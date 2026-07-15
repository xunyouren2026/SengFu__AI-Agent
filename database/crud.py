"""
AGI Unified Framework - CRUD 操作模块

本模块提供所有数据库模型的 CRUD（创建、读取、更新、删除）操作类。
每个 CRUD 类封装了对应模型的数据库操作，提供类型安全的接口。

设计原则:
    1. 所有方法接受 Session 参数，支持依赖注入
    2. 完整的类型注解和文档字符串
    3. 支持分页、排序、过滤
    4. 异常处理和日志记录
    5. 支持批量操作

依赖:
    - sqlalchemy >= 1.4 (可选，优雅降级)
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)

# 尝试导入 SQLAlchemy
try:
    from sqlalchemy import or_, and_, desc, asc, func, select, text
    from sqlalchemy.orm import Session
    from sqlalchemy.exc import SQLAlchemyError, IntegrityError

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy 未安装。CRUD 操作将不可用。")

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
        ModelStatus,
        ModelProvider,
        ModelType,
        TrainingStatus,
        ContentType,
        WorkflowStatus,
        ExecutionStatus,
        AgentStatus,
        PluginStatus,
        ChannelType,
        ChannelStatus,
        DatasetType,
        AuditAction,
        ResourceType,
        LoadBalanceStrategy,
    )
    from database.utils import hash_password, verify_password, paginate_query
except ImportError:
    # 优雅降级
    pass


# ============================================================
# 基础 CRUD 类
# ============================================================

class BaseCRUD:
    """
    基础 CRUD 操作类

    提供通用的数据库操作方法，所有具体 CRUD 类继承此类。

    属性:
        model: SQLAlchemy ORM 模型类
    """

    def __init__(self, model: Any) -> None:
        """
        初始化 CRUD 类

        参数:
            model: SQLAlchemy ORM 模型类
        """
        self.model = model

    def get_by_id(self, db: "Session", id: int) -> Optional[Any]:
        """
        根据 ID 获取记录

        参数:
            db: 数据库会话
            id: 记录 ID

        返回:
            模型实例或 None
        """
        try:
            return db.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            logger.error(f"获取 {self.model.__name__} ID={id} 失败: {e}")
            return None

    def get_all(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        order_desc: bool = True,
    ) -> List[Any]:
        """
        获取所有记录（分页）

        参数:
            db: 数据库会话
            skip: 跳过记录数
            limit: 返回记录数
            order_by: 排序字段名
            order_desc: 是否降序

        返回:
            模型实例列表
        """
        try:
            query = db.query(self.model)
            if order_by and hasattr(self.model, order_by):
                column = getattr(self.model, order_by)
                query = query.order_by(desc(column) if order_desc else asc(column))
            return query.offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"获取 {self.model.__name__} 列表失败: {e}")
            return []

    def count(self, db: "Session") -> int:
        """
        获取记录总数

        参数:
            db: 数据库会话

        返回:
            记录总数
        """
        try:
            return db.query(func.count(self.model.id)).scalar() or 0
        except SQLAlchemyError as e:
            logger.error(f"统计 {self.model.__name__} 数量失败: {e}")
            return 0

    def exists(self, db: "Session", id: int) -> bool:
        """
        检查记录是否存在

        参数:
            db: 数据库会话
            id: 记录 ID

        返回:
            是否存在
        """
        try:
            return (
                db.query(func.count(self.model.id))
                .filter(self.model.id == id)
                .scalar() or 0
            ) > 0
        except SQLAlchemyError as e:
            logger.error(f"检查 {self.model.__name__} 存在性失败: {e}")
            return False

    def delete_by_id(self, db: "Session", id: int) -> bool:
        """
        根据 ID 删除记录

        参数:
            db: 数据库会话
            id: 记录 ID

        返回:
            是否删除成功
        """
        try:
            obj = db.query(self.model).filter(self.model.id == id).first()
            if obj:
                db.delete(obj)
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"删除 {self.model.__name__} ID={id} 失败: {e}")
            return False

    def search(
        self,
        db: "Session",
        search_term: str,
        search_fields: List[str],
        skip: int = 0,
        limit: int = 100,
    ) -> List[Any]:
        """
        搜索记录

        参数:
            db: 数据库会话
            search_term: 搜索关键词
            search_fields: 搜索字段列表
            skip: 跳过记录数
            limit: 返回记录数

        返回:
            匹配的模型实例列表
        """
        try:
            conditions = []
            for field in search_fields:
                if hasattr(self.model, field):
                    column = getattr(self.model, field)
                    conditions.append(column.ilike(f"%{search_term}%"))

            if conditions:
                query = db.query(self.model).filter(or_(*conditions))
                return query.offset(skip).limit(limit).all()
            return []
        except SQLAlchemyError as e:
            logger.error(f"搜索 {self.model.__name__} 失败: {e}")
            return []


# ============================================================
# 用户 CRUD
# ============================================================

class UserCRUD(BaseCRUD):
    """
    用户 CRUD 操作类

    提供用户的创建、查询、更新、删除、认证等操作。
    """

    def __init__(self) -> None:
        super().__init__(User)

    def create(
        self,
        db: "Session",
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
        **kwargs: Any,
    ) -> Optional[User]:
        """
        创建新用户

        参数:
            db: 数据库会话
            username: 用户名
            email: 电子邮件
            password: 明文密码
            role: 用户角色
            **kwargs: 额外字段

        返回:
            创建的用户实例或 None
        """
        try:
            user = User(
                username=username,
                email=email,
                password_hash=hash_password(password),
                role=role,
                **kwargs,
            )
            db.add(user)
            db.flush()

            # 创建默认用户设置
            settings = UserSettings(user_id=user.id)
            db.add(settings)

            db.commit()
            db.refresh(user)
            logger.info(f"用户创建成功: {username} (ID={user.id})")
            return user
        except IntegrityError as e:
            db.rollback()
            logger.warning(f"用户创建失败（已存在）: {e}")
            return None
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"用户创建失败: {e}")
            return None

    def get_by_username(self, db: "Session", username: str) -> Optional[User]:
        """根据用户名获取用户"""
        try:
            return db.query(User).filter(User.username == username).first()
        except SQLAlchemyError as e:
            logger.error(f"根据用户名获取用户失败: {e}")
            return None

    def get_by_email(self, db: "Session", email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        try:
            return db.query(User).filter(User.email == email).first()
        except SQLAlchemyError as e:
            logger.error(f"根据邮箱获取用户失败: {e}")
            return None

    def authenticate(
        self, db: "Session", username: str, password: str
    ) -> Optional[User]:
        """
        用户认证

        参数:
            db: 数据库会话
            username: 用户名
            password: 明文密码

        返回:
            认证成功返回用户实例，否则返回 None
        """
        try:
            user = self.get_by_username(db, username)
            if user and verify_password(password, user.password_hash):
                if user.is_locked:
                    logger.warning(f"用户已锁定: {username}")
                    return None
                # 更新登录信息
                user.last_login = datetime.now(timezone.utc)
                user.login_count = (user.login_count or 0) + 1
                user.failed_login_attempts = 0
                db.commit()
                db.refresh(user)
                return user
            elif user:
                # 增加失败计数
                user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
                db.commit()
            return None
        except SQLAlchemyError as e:
            logger.error(f"用户认证失败: {e}")
            return None

    def update(
        self, db: "Session", user_id: int, **kwargs: Any
    ) -> Optional[User]:
        """更新用户信息"""
        try:
            user = self.get_by_id(db, user_id)
            if not user:
                return None

            # 不允许直接更新密码哈希
            if "password_hash" in kwargs:
                del kwargs["password_hash"]
            if "password" in kwargs:
                kwargs["password_hash"] = hash_password(kwargs.pop("password"))
                kwargs["password_changed_at"] = datetime.now(timezone.utc)

            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)

            db.commit()
            db.refresh(user)
            return user
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新用户 ID={user_id} 失败: {e}")
            return None

    def update_password(
        self, db: "Session", user_id: int, new_password: str
    ) -> bool:
        """更新用户密码"""
        try:
            user = self.get_by_id(db, user_id)
            if not user:
                return False
            user.password_hash = hash_password(new_password)
            user.password_changed_at = datetime.now(timezone.utc)
            user.failed_login_attempts = 0
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新密码失败: {e}")
            return False

    def list_users(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 100,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[User], int]:
        """
        列出用户（支持过滤和搜索）

        返回:
            (用户列表, 总数)
        """
        try:
            query = db.query(User)
            count_query = db.query(func.count(User.id))

            if role is not None:
                query = query.filter(User.role == role)
                count_query = count_query.filter(User.role == role)
            if is_active is not None:
                query = query.filter(User.is_active == is_active)
                count_query = count_query.filter(User.is_active == is_active)
            if search:
                search_filter = or_(
                    User.username.ilike(f"%{search}%"),
                    User.email.ilike(f"%{search}%"),
                    User.display_name.ilike(f"%{search}%"),
                )
                query = query.filter(search_filter)
                count_query = count_query.filter(search_filter)

            total = count_query.scalar() or 0
            users = query.order_by(desc(User.created_at)).offset(skip).limit(limit).all()
            return users, total
        except SQLAlchemyError as e:
            logger.error(f"列出用户失败: {e}")
            return [], 0

    def lock_user(self, db: "Session", user_id: int) -> bool:
        """锁定用户"""
        try:
            user = self.get_by_id(db, user_id)
            if user:
                user.is_locked = True
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"锁定用户失败: {e}")
            return False

    def unlock_user(self, db: "Session", user_id: int) -> bool:
        """解锁用户"""
        try:
            user = self.get_by_id(db, user_id)
            if user:
                user.is_locked = False
                user.failed_login_attempts = 0
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"解锁用户失败: {e}")
            return False

    def get_settings(self, db: "Session", user_id: int) -> Optional[UserSettings]:
        """获取用户设置"""
        try:
            return (
                db.query(UserSettings)
                .filter(UserSettings.user_id == user_id)
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取用户设置失败: {e}")
            return None

    def update_settings(
        self, db: "Session", user_id: int, **kwargs: Any
    ) -> Optional[UserSettings]:
        """更新用户设置"""
        try:
            settings = self.get_settings(db, user_id)
            if not settings:
                settings = UserSettings(user_id=user_id)
                db.add(settings)

            for key, value in kwargs.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)

            db.commit()
            db.refresh(settings)
            return settings
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新用户设置失败: {e}")
            return None


# ============================================================
# 对话 CRUD
# ============================================================

class ConversationCRUD(BaseCRUD):
    """
    对话 CRUD 操作类

    提供对话的创建、查询、更新、删除和搜索操作。
    """

    def __init__(self) -> None:
        super().__init__(Conversation)

    def create(
        self,
        db: "Session",
        user_id: int,
        title: str = "新对话",
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Conversation]:
        """创建新对话"""
        try:
            conversation = Conversation(
                user_id=user_id,
                title=title,
                model_name=model_name,
                system_prompt=system_prompt,
                **kwargs,
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            logger.info(f"对话创建成功: {title} (ID={conversation.id})")
            return conversation
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建对话失败: {e}")
            return None

    def get_by_user(
        self,
        db: "Session",
        user_id: int,
        skip: int = 0,
        limit: int = 50,
        include_archived: bool = False,
    ) -> Tuple[List[Conversation], int]:
        """获取用户的对话列表"""
        try:
            query = db.query(Conversation).filter(Conversation.user_id == user_id)
            count_query = db.query(func.count(Conversation.id)).filter(
                Conversation.user_id == user_id
            )

            if not include_archived:
                query = query.filter(Conversation.is_archived == False)  # noqa: E712
                count_query = count_query.filter(Conversation.is_archived == False)  # noqa: E712

            total = count_query.scalar() or 0
            conversations = (
                query.order_by(desc(Conversation.updated_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
            return conversations, total
        except SQLAlchemyError as e:
            logger.error(f"获取用户对话列表失败: {e}")
            return [], 0

    def update(
        self, db: "Session", conversation_id: int, **kwargs: Any
    ) -> Optional[Conversation]:
        """更新对话信息"""
        try:
            conversation = self.get_by_id(db, conversation_id)
            if not conversation:
                return None

            for key, value in kwargs.items():
                if hasattr(conversation, key):
                    setattr(conversation, key, value)

            db.commit()
            db.refresh(conversation)
            return conversation
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新对话失败: {e}")
            return None

    def archive(self, db: "Session", conversation_id: int) -> bool:
        """归档对话"""
        return self._set_field(db, conversation_id, "is_archived", True)

    def unarchive(self, db: "Session", conversation_id: int) -> bool:
        """取消归档对话"""
        return self._set_field(db, conversation_id, "is_archived", False)

    def pin(self, db: "Session", conversation_id: int) -> bool:
        """置顶对话"""
        return self._set_field(db, conversation_id, "is_pinned", True)

    def unpin(self, db: "Session", conversation_id: int) -> bool:
        """取消置顶对话"""
        return self._set_field(db, conversation_id, "is_pinned", False)

    def _set_field(
        self, db: "Session", conversation_id: int, field: str, value: Any
    ) -> bool:
        """设置对话字段值"""
        try:
            conversation = self.get_by_id(db, conversation_id)
            if conversation:
                setattr(conversation, field, value)
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"设置对话字段失败: {e}")
            return False

    def search(
        self,
        db: "Session",
        user_id: int,
        search_term: str,
        skip: int = 0,
        limit: int = 50,
    ) -> List[Conversation]:
        """搜索用户对话"""
        try:
            query = db.query(Conversation).filter(
                Conversation.user_id == user_id,
                or_(
                    Conversation.title.ilike(f"%{search_term}%"),
                    Conversation.description.ilike(f"%{search_term}%"),
                ),
            )
            return query.order_by(desc(Conversation.updated_at)).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"搜索对话失败: {e}")
            return []

    def get_stats(self, db: "Session", user_id: int) -> Dict[str, Any]:
        """获取用户对话统计"""
        try:
            total = (
                db.query(func.count(Conversation.id))
                .filter(Conversation.user_id == user_id)
                .scalar() or 0
            )
            archived = (
                db.query(func.count(Conversation.id))
                .filter(
                    Conversation.user_id == user_id,
                    Conversation.is_archived == True,  # noqa: E712
                )
                .scalar() or 0
            )
            pinned = (
                db.query(func.count(Conversation.id))
                .filter(
                    Conversation.user_id == user_id,
                    Conversation.is_pinned == True,  # noqa: E712
                )
                .scalar() or 0
            )
            total_tokens = (
                db.query(func.sum(Conversation.total_tokens))
                .filter(Conversation.user_id == user_id)
                .scalar() or 0
            )
            return {
                "total": total,
                "archived": archived,
                "pinned": pinned,
                "total_tokens": total_tokens,
            }
        except SQLAlchemyError as e:
            logger.error(f"获取对话统计失败: {e}")
            return {"total": 0, "archived": 0, "pinned": 0, "total_tokens": 0}

    def delete_by_user(self, db: "Session", user_id: int) -> int:
        """删除用户的所有对话"""
        try:
            count = (
                db.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .delete()
            )
            db.commit()
            return count
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"删除用户对话失败: {e}")
            return 0


# ============================================================
# 消息 CRUD
# ============================================================

class MessageCRUD(BaseCRUD):
    """
    消息 CRUD 操作类

    提供消息的创建、查询和删除操作。
    """

    def __init__(self) -> None:
        super().__init__(Message)

    def create(
        self,
        db: "Session",
        conversation_id: int,
        role: str,
        content: str,
        **kwargs: Any,
    ) -> Optional[Message]:
        """创建新消息"""
        try:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                **kwargs,
            )
            db.add(message)

            # 更新对话统计
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            if conversation:
                conversation.message_count = (conversation.message_count or 0) + 1
                if message.total_tokens:
                    conversation.total_tokens = (
                        (conversation.total_tokens or 0) + message.total_tokens
                    )
                conversation.last_message_at = datetime.now(timezone.utc)
                conversation.updated_at = datetime.now(timezone.utc)

            db.commit()
            db.refresh(message)
            return message
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建消息失败: {e}")
            return None

    def get_by_conversation(
        self,
        db: "Session",
        conversation_id: int,
        skip: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> List[Message]:
        """获取对话的消息列表"""
        try:
            query = db.query(Message).filter(
                Message.conversation_id == conversation_id
            )
            if not include_deleted:
                query = query.filter(Message.is_deleted == False)  # noqa: E712
            return query.order_by(asc(Message.created_at)).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"获取对话消息失败: {e}")
            return []

    def soft_delete(self, db: "Session", message_id: int) -> bool:
        """软删除消息"""
        try:
            message = self.get_by_id(db, message_id)
            if message:
                message.is_deleted = True
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"软删除消息失败: {e}")
            return False

    def update(
        self, db: "Session", message_id: int, **kwargs: Any
    ) -> Optional[Message]:
        """更新消息"""
        try:
            message = self.get_by_id(db, message_id)
            if not message:
                return None
            if "content" in kwargs:
                message.is_edited = True
            for key, value in kwargs.items():
                if hasattr(message, key):
                    setattr(message, key, value)
            db.commit()
            db.refresh(message)
            return message
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新消息失败: {e}")
            return None

    def rate_message(
        self, db: "Session", message_id: int, rating: int, feedback: Optional[str] = None
    ) -> bool:
        """对消息评分"""
        return bool(
            self.update(db, message_id, rating=rating, feedback=feedback)
        )

    def get_conversation_stats(
        self, db: "Session", conversation_id: int
    ) -> Dict[str, Any]:
        """获取对话的消息统计"""
        try:
            total = (
                db.query(func.count(Message.id))
                .filter(
                    Message.conversation_id == conversation_id,
                    Message.is_deleted == False,  # noqa: E712
                )
                .scalar() or 0
            )
            user_messages = (
                db.query(func.count(Message.id))
                .filter(
                    Message.conversation_id == conversation_id,
                    Message.role == "user",
                    Message.is_deleted == False,  # noqa: E712
                )
                .scalar() or 0
            )
            assistant_messages = (
                db.query(func.count(Message.id))
                .filter(
                    Message.conversation_id == conversation_id,
                    Message.role == "assistant",
                    Message.is_deleted == False,  # noqa: E712
                )
                .scalar() or 0
            )
            total_tokens = (
                db.query(func.sum(Message.total_tokens))
                .filter(
                    Message.conversation_id == conversation_id,
                    Message.is_deleted == False,  # noqa: E712
                )
                .scalar() or 0
            )
            return {
                "total": total,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "total_tokens": total_tokens,
            }
        except SQLAlchemyError as e:
            logger.error(f"获取消息统计失败: {e}")
            return {"total": 0, "user_messages": 0, "assistant_messages": 0, "total_tokens": 0}

    def delete_by_conversation(self, db: "Session", conversation_id: int) -> int:
        """删除对话的所有消息"""
        try:
            count = (
                db.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .delete()
            )
            db.commit()
            return count
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"删除对话消息失败: {e}")
            return 0


# ============================================================
# 模型 CRUD
# ============================================================

class ModelCRUD(BaseCRUD):
    """
    模型 CRUD 操作类

    提供模型配置的创建、查询、更新、删除和连接测试操作。
    """

    def __init__(self) -> None:
        super().__init__(Model)

    def create(
        self,
        db: "Session",
        name: str,
        provider: ModelProvider,
        model_type: ModelType = ModelType.LLM,
        **kwargs: Any,
    ) -> Optional[Model]:
        """创建新模型配置"""
        try:
            model = Model(
                name=name,
                provider=provider,
                model_type=model_type,
                **kwargs,
            )
            db.add(model)
            db.commit()
            db.refresh(model)
            logger.info(f"模型配置创建成功: {name} (ID={model.id})")
            return model
        except IntegrityError as e:
            db.rollback()
            logger.warning(f"模型配置创建失败（名称已存在）: {e}")
            return None
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建模型配置失败: {e}")
            return None

    def get_by_name(self, db: "Session", name: str) -> Optional[Model]:
        """根据名称获取模型"""
        try:
            return db.query(Model).filter(Model.name == name).first()
        except SQLAlchemyError as e:
            logger.error(f"根据名称获取模型失败: {e}")
            return None

    def get_by_provider(
        self,
        db: "Session",
        provider: ModelProvider,
        status: Optional[ModelStatus] = None,
    ) -> List[Model]:
        """根据提供商获取模型列表"""
        try:
            query = db.query(Model).filter(Model.provider == provider)
            if status is not None:
                query = query.filter(Model.status == status)
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"根据提供商获取模型失败: {e}")
            return []

    def get_active_models(self, db: "Session") -> List[Model]:
        """获取所有活跃模型"""
        try:
            return (
                db.query(Model)
                .filter(Model.status == ModelStatus.ACTIVE)
                .order_by(desc(Model.priority))
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取活跃模型失败: {e}")
            return []

    def get_default_model(self, db: "Session") -> Optional[Model]:
        """获取默认模型"""
        try:
            return (
                db.query(Model)
                .filter(
                    Model.is_default == True,  # noqa: E712
                    Model.status == ModelStatus.ACTIVE,
                )
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取默认模型失败: {e}")
            return None

    def update(
        self, db: "Session", model_id: int, **kwargs: Any
    ) -> Optional[Model]:
        """更新模型配置"""
        try:
            model = self.get_by_id(db, model_id)
            if not model:
                return None
            for key, value in kwargs.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            db.commit()
            db.refresh(model)
            return model
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新模型配置失败: {e}")
            return None

    def set_status(
        self, db: "Session", model_id: int, status: ModelStatus
    ) -> bool:
        """设置模型状态"""
        return bool(self.update(db, model_id, status=status))

    def set_default(self, db: "Session", model_id: int) -> bool:
        """设置为默认模型"""
        try:
            # 先取消所有默认
            db.query(Model).filter(Model.is_default == True).update(  # noqa: E712
                {"is_default": False}
            )
            # 设置新默认
            model = self.get_by_id(db, model_id)
            if model:
                model.is_default = True
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"设置默认模型失败: {e}")
            return False

    def list_models(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 100,
        provider: Optional[ModelProvider] = None,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Model], int]:
        """列出模型（支持过滤和搜索）"""
        try:
            query = db.query(Model)
            count_query = db.query(func.count(Model.id))

            if provider is not None:
                query = query.filter(Model.provider == provider)
                count_query = count_query.filter(Model.provider == provider)
            if model_type is not None:
                query = query.filter(Model.model_type == model_type)
                count_query = count_query.filter(Model.model_type == model_type)
            if status is not None:
                query = query.filter(Model.status == status)
                count_query = count_query.filter(Model.status == status)
            if search:
                search_filter = or_(
                    Model.name.ilike(f"%{search}%"),
                    Model.display_name.ilike(f"%{search}%"),
                    Model.description.ilike(f"%{search}%"),
                )
                query = query.filter(search_filter)
                count_query = count_query.filter(search_filter)

            total = count_query.scalar() or 0
            models = query.order_by(desc(Model.priority)).offset(skip).limit(limit).all()
            return models, total
        except SQLAlchemyError as e:
            logger.error(f"列出模型失败: {e}")
            return [], 0

    def update_usage_stats(
        self,
        db: "Session",
        model_id: int,
        tokens_used: int = 0,
        latency_ms: Optional[float] = None,
        is_error: bool = False,
    ) -> bool:
        """更新模型使用统计"""
        try:
            model = self.get_by_id(db, model_id)
            if not model:
                return False
            model.total_requests = (model.total_requests or 0) + 1
            model.total_tokens_used = (model.total_tokens_used or 0) + tokens_used
            if is_error:
                model.total_errors = (model.total_errors or 0) + 1
            if latency_ms is not None:
                # 简单移动平均
                old_avg = model.avg_latency_ms or latency_ms
                count = model.total_requests or 1
                model.avg_latency_ms = old_avg + (latency_ms - old_avg) / count
            model.last_used_at = datetime.now(timezone.utc)
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新模型统计失败: {e}")
            return False

    def test_connection(self, db: "Session", model_id: int) -> Dict[str, Any]:
        """
        测试模型连接

        返回:
            测试结果字典: {"success": bool, "latency_ms": float, "error": str}
        """
        import time
        try:
            model = self.get_by_id(db, model_id)
            if not model:
                return {"success": False, "latency_ms": 0, "error": "模型不存在"}

            if not model.api_endpoint:
                return {"success": False, "latency_ms": 0, "error": "未配置API端点"}

            # 模拟连接测试
            start_time = time.time()
            # 实际实现中这里会发起 HTTP 请求
            time.sleep(0.1)  # 模拟延迟
            latency = (time.time() - start_time) * 1000

            model.last_health_check = datetime.now(timezone.utc)
            db.commit()

            return {"success": True, "latency_ms": round(latency, 2), "error": None}
        except Exception as e:
            return {"success": False, "latency_ms": 0, "error": str(e)}


# ============================================================
# 训练任务 CRUD
# ============================================================

class TrainingJobCRUD(BaseCRUD):
    """
    训练任务 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(TrainingJob)

    def create(
        self,
        db: "Session",
        name: str,
        user_id: Optional[int] = None,
        model_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[TrainingJob]:
        """创建训练任务"""
        try:
            job = TrainingJob(
                name=name,
                user_id=user_id,
                model_id=model_id,
                **kwargs,
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            logger.info(f"训练任务创建成功: {name} (ID={job.id})")
            return job
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建训练任务失败: {e}")
            return None

    def update_status(
        self,
        db: "Session",
        job_id: int,
        status: TrainingStatus,
        **kwargs: Any,
    ) -> Optional[TrainingJob]:
        """更新训练任务状态"""
        try:
            job = self.get_by_id(db, job_id)
            if not job:
                return None

            job.status = status

            if status == TrainingStatus.RUNNING and not job.started_at:
                job.started_at = datetime.now(timezone.utc)
            elif status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
                job.completed_at = datetime.now(timezone.utc)

            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)

            db.commit()
            db.refresh(job)
            return job
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新训练任务状态失败: {e}")
            return None

    def update_progress(
        self,
        db: "Session",
        job_id: int,
        progress: float,
        current_epoch: Optional[int] = None,
        current_step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新训练进度"""
        try:
            job = self.get_by_id(db, job_id)
            if not job:
                return False
            job.progress = progress
            if current_epoch is not None:
                job.current_epoch = current_epoch
            if current_step is not None:
                job.current_step = current_step
            if metrics is not None:
                job.metrics_json = metrics
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新训练进度失败: {e}")
            return False

    def list_jobs(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 50,
        user_id: Optional[int] = None,
        status: Optional[TrainingStatus] = None,
    ) -> Tuple[List[TrainingJob], int]:
        """列出训练任务"""
        try:
            query = db.query(TrainingJob)
            count_query = db.query(func.count(TrainingJob.id))

            if user_id is not None:
                query = query.filter(TrainingJob.user_id == user_id)
                count_query = count_query.filter(TrainingJob.user_id == user_id)
            if status is not None:
                query = query.filter(TrainingJob.status == status)
                count_query = count_query.filter(TrainingJob.status == status)

            total = count_query.scalar() or 0
            jobs = query.order_by(desc(TrainingJob.created_at)).offset(skip).limit(limit).all()
            return jobs, total
        except SQLAlchemyError as e:
            logger.error(f"列出训练任务失败: {e}")
            return [], 0

    def get_running_jobs(self, db: "Session") -> List[TrainingJob]:
        """获取正在运行的训练任务"""
        try:
            return (
                db.query(TrainingJob)
                .filter(TrainingJob.status == TrainingStatus.RUNNING)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取运行中任务失败: {e}")
            return []

    def cancel_job(self, db: "Session", job_id: int) -> bool:
        """取消训练任务"""
        try:
            job = self.get_by_id(db, job_id)
            if job and job.status in (
                TrainingStatus.PENDING,
                TrainingStatus.QUEUED,
                TrainingStatus.RUNNING,
                TrainingStatus.PAUSED,
            ):
                job.status = TrainingStatus.CANCELLED
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"取消训练任务失败: {e}")
            return False


# ============================================================
# 检查点 CRUD
# ============================================================

class CheckpointCRUD(BaseCRUD):
    """
    检查点 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(Checkpoint)

    def create(
        self,
        db: "Session",
        training_job_id: int,
        epoch: int,
        step: int = 0,
        **kwargs: Any,
    ) -> Optional[Checkpoint]:
        """创建检查点"""
        try:
            checkpoint = Checkpoint(
                training_job_id=training_job_id,
                epoch=epoch,
                step=step,
                **kwargs,
            )
            db.add(checkpoint)
            db.commit()
            db.refresh(checkpoint)
            return checkpoint
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建检查点失败: {e}")
            return None

    def get_by_job(
        self, db: "Session", training_job_id: int
    ) -> List[Checkpoint]:
        """获取训练任务的所有检查点"""
        try:
            return (
                db.query(Checkpoint)
                .filter(Checkpoint.training_job_id == training_job_id)
                .order_by(desc(Checkpoint.epoch), desc(Checkpoint.step))
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取检查点列表失败: {e}")
            return []

    def get_best_checkpoint(
        self, db: "Session", training_job_id: int
    ) -> Optional[Checkpoint]:
        """获取最佳检查点"""
        try:
            return (
                db.query(Checkpoint)
                .filter(
                    Checkpoint.training_job_id == training_job_id,
                    Checkpoint.is_best == True,  # noqa: E712
                )
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取最佳检查点失败: {e}")
            return None

    def mark_as_best(self, db: "Session", checkpoint_id: int) -> bool:
        """标记为最佳检查点"""
        try:
            checkpoint = self.get_by_id(db, checkpoint_id)
            if not checkpoint:
                return False
            # 取消同任务的其他最佳标记
            db.query(Checkpoint).filter(
                Checkpoint.training_job_id == checkpoint.training_job_id,
                Checkpoint.is_best == True,  # noqa: E712
            ).update({"is_best": False})
            checkpoint.is_best = True
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"标记最佳检查点失败: {e}")
            return False


# ============================================================
# 生成内容 CRUD
# ============================================================

class GeneratedContentCRUD(BaseCRUD):
    """
    生成内容 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(GeneratedContent)

    def create(
        self,
        db: "Session",
        user_id: int,
        content_type: ContentType,
        prompt: str,
        **kwargs: Any,
    ) -> Optional[GeneratedContent]:
        """创建生成内容记录"""
        try:
            content = GeneratedContent(
                user_id=user_id,
                type=content_type,
                prompt=prompt,
                **kwargs,
            )
            db.add(content)
            db.commit()
            db.refresh(content)
            return content
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建生成内容记录失败: {e}")
            return None

    def get_by_user(
        self,
        db: "Session",
        user_id: int,
        content_type: Optional[ContentType] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[GeneratedContent], int]:
        """获取用户的生成内容"""
        try:
            query = db.query(GeneratedContent).filter(
                GeneratedContent.user_id == user_id
            )
            count_query = db.query(func.count(GeneratedContent.id)).filter(
                GeneratedContent.user_id == user_id
            )
            if content_type is not None:
                query = query.filter(GeneratedContent.type == content_type)
                count_query = count_query.filter(GeneratedContent.type == content_type)

            total = count_query.scalar() or 0
            contents = query.order_by(desc(GeneratedContent.created_at)).offset(skip).limit(limit).all()
            return contents, total
        except SQLAlchemyError as e:
            logger.error(f"获取生成内容失败: {e}")
            return [], 0

    def get_by_type(
        self,
        db: "Session",
        content_type: ContentType,
        skip: int = 0,
        limit: int = 50,
    ) -> List[GeneratedContent]:
        """按类型获取生成内容"""
        try:
            return (
                db.query(GeneratedContent)
                .filter(GeneratedContent.type == content_type)
                .order_by(desc(GeneratedContent.created_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"按类型获取生成内容失败: {e}")
            return []

    def toggle_favorite(self, db: "Session", content_id: int) -> bool:
        """切换收藏状态"""
        try:
            content = self.get_by_id(db, content_id)
            if content:
                content.is_favorite = not content.is_favorite
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"切换收藏状态失败: {e}")
            return False

    def list_public(
        self,
        db: "Session",
        content_type: Optional[ContentType] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[GeneratedContent]:
        """列出公开的生成内容"""
        try:
            query = db.query(GeneratedContent).filter(
                GeneratedContent.is_public == True  # noqa: E712
            )
            if content_type is not None:
                query = query.filter(GeneratedContent.type == content_type)
            return query.order_by(desc(GeneratedContent.created_at)).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"列出公开内容失败: {e}")
            return []


# ============================================================
# 工作流 CRUD
# ============================================================

class WorkflowCRUD(BaseCRUD):
    """
    工作流 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(Workflow)

    def create(
        self,
        db: "Session",
        name: str,
        user_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[Workflow]:
        """创建工作流"""
        try:
            workflow = Workflow(name=name, user_id=user_id, **kwargs)
            db.add(workflow)
            db.commit()
            db.refresh(workflow)
            logger.info(f"工作流创建成功: {name} (ID={workflow.id})")
            return workflow
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建工作流失败: {e}")
            return None

    def get_by_user(
        self,
        db: "Session",
        user_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[Workflow], int]:
        """获取用户的工作流"""
        try:
            query = db.query(Workflow).filter(Workflow.user_id == user_id)
            count_query = db.query(func.count(Workflow.id)).filter(
                Workflow.user_id == user_id
            )
            total = count_query.scalar() or 0
            workflows = (
                query.order_by(desc(Workflow.updated_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
            return workflows, total
        except SQLAlchemyError as e:
            logger.error(f"获取用户工作流失败: {e}")
            return [], 0

    def get_templates(self, db: "Session") -> List[Workflow]:
        """获取工作流模板"""
        try:
            return (
                db.query(Workflow)
                .filter(Workflow.is_template == True)  # noqa: E712
                .order_by(asc(Workflow.name))
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取工作流模板失败: {e}")
            return []

    def update(
        self, db: "Session", workflow_id: int, **kwargs: Any
    ) -> Optional[Workflow]:
        """更新工作流"""
        try:
            workflow = self.get_by_id(db, workflow_id)
            if not workflow:
                return None
            for key, value in kwargs.items():
                if hasattr(workflow, key):
                    setattr(workflow, key, value)
            db.commit()
            db.refresh(workflow)
            return workflow
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新工作流失败: {e}")
            return None

    def duplicate(
        self, db: "Session", workflow_id: int, new_name: Optional[str] = None
    ) -> Optional[Workflow]:
        """复制工作流"""
        try:
            original = self.get_by_id(db, workflow_id)
            if not original:
                return None
            new_workflow = Workflow(
                name=new_name or f"{original.name} (副本)",
                user_id=original.user_id,
                description=original.description,
                config_json=original.config_json,
                graph_json=original.graph_json,
                variables=original.variables,
                triggers=original.triggers,
                status=WorkflowStatus.DRAFT,
                category=original.category,
                tags=original.tags,
            )
            db.add(new_workflow)
            db.commit()
            db.refresh(new_workflow)
            return new_workflow
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"复制工作流失败: {e}")
            return None

    def list_workflows(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 50,
        status: Optional[WorkflowStatus] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Workflow], int]:
        """列出工作流"""
        try:
            query = db.query(Workflow)
            count_query = db.query(func.count(Workflow.id))

            if status is not None:
                query = query.filter(Workflow.status == status)
                count_query = count_query.filter(Workflow.status == status)
            if category:
                query = query.filter(Workflow.category == category)
                count_query = count_query.filter(Workflow.category == category)
            if search:
                search_filter = or_(
                    Workflow.name.ilike(f"%{search}%"),
                    Workflow.description.ilike(f"%{search}%"),
                )
                query = query.filter(search_filter)
                count_query = count_query.filter(search_filter)

            total = count_query.scalar() or 0
            workflows = query.order_by(desc(Workflow.updated_at)).offset(skip).limit(limit).all()
            return workflows, total
        except SQLAlchemyError as e:
            logger.error(f"列出工作流失败: {e}")
            return [], 0


# ============================================================
# 工作流执行 CRUD
# ============================================================

class WorkflowExecutionCRUD(BaseCRUD):
    """
    工作流执行记录 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(WorkflowExecution)

    def create(
        self,
        db: "Session",
        workflow_id: int,
        **kwargs: Any,
    ) -> Optional[WorkflowExecution]:
        """创建执行记录"""
        try:
            execution = WorkflowExecution(workflow_id=workflow_id, **kwargs)
            db.add(execution)
            db.commit()
            db.refresh(execution)
            return execution
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建执行记录失败: {e}")
            return None

    def start_execution(
        self, db: "Session", execution_id: int
    ) -> Optional[WorkflowExecution]:
        """开始执行"""
        try:
            execution = self.get_by_id(db, execution_id)
            if not execution:
                return None
            execution.status = ExecutionStatus.RUNNING
            execution.started_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(execution)
            return execution
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"开始执行失败: {e}")
            return None

    def complete_execution(
        self,
        db: "Session",
        execution_id: int,
        result: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkflowExecution]:
        """完成执行"""
        try:
            execution = self.get_by_id(db, execution_id)
            if not execution:
                return None
            execution.status = ExecutionStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            if execution.started_at:
                execution.duration_ms = int(
                    (execution.completed_at - execution.started_at).total_seconds() * 1000
                )
            if result:
                execution.result_json = result
            if output:
                execution.output_json = output
            db.commit()
            db.refresh(execution)
            return execution
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"完成执行失败: {e}")
            return None

    def fail_execution(
        self,
        db: "Session",
        execution_id: int,
        error_message: str,
        error_traceback: Optional[str] = None,
    ) -> Optional[WorkflowExecution]:
        """标记执行失败"""
        try:
            execution = self.get_by_id(db, execution_id)
            if not execution:
                return None
            execution.status = ExecutionStatus.FAILED
            execution.completed_at = datetime.now(timezone.utc)
            execution.error_message = error_message
            execution.error_traceback = error_traceback
            if execution.started_at:
                execution.duration_ms = int(
                    (execution.completed_at - execution.started_at).total_seconds() * 1000
                )
            db.commit()
            db.refresh(execution)
            return execution
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"标记执行失败: {e}")
            return None

    def get_by_workflow(
        self,
        db: "Session",
        workflow_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> List[WorkflowExecution]:
        """获取工作流的执行记录"""
        try:
            return (
                db.query(WorkflowExecution)
                .filter(WorkflowExecution.workflow_id == workflow_id)
                .order_by(desc(WorkflowExecution.created_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取执行记录失败: {e}")
            return []


# ============================================================
# 智能体 CRUD
# ============================================================

class AgentCRUD(BaseCRUD):
    """
    智能体 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(Agent)

    def create(
        self,
        db: "Session",
        name: str,
        user_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[Agent]:
        """创建智能体"""
        try:
            agent = Agent(name=name, user_id=user_id, **kwargs)
            db.add(agent)
            db.commit()
            db.refresh(agent)
            logger.info(f"智能体创建成功: {name} (ID={agent.id})")
            return agent
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建智能体失败: {e}")
            return None

    def get_by_user(
        self,
        db: "Session",
        user_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[Agent], int]:
        """获取用户的智能体"""
        try:
            query = db.query(Agent).filter(Agent.user_id == user_id)
            count_query = db.query(func.count(Agent.id)).filter(
                Agent.user_id == user_id
            )
            total = count_query.scalar() or 0
            agents = query.order_by(desc(Agent.updated_at)).offset(skip).limit(limit).all()
            return agents, total
        except SQLAlchemyError as e:
            logger.error(f"获取用户智能体失败: {e}")
            return [], 0

    def get_public_agents(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 50,
    ) -> List[Agent]:
        """获取公开的智能体"""
        try:
            return (
                db.query(Agent)
                .filter(
                    Agent.is_public == True,  # noqa: E712
                    Agent.status == AgentStatus.ACTIVE,
                )
                .order_by(desc(Agent.total_interactions))
                .offset(skip)
                .limit(limit)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取公开智能体失败: {e}")
            return []

    def get_templates(self, db: "Session") -> List[Agent]:
        """获取智能体模板"""
        try:
            return (
                db.query(Agent)
                .filter(Agent.is_template == True)  # noqa: E712
                .order_by(asc(Agent.name))
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取智能体模板失败: {e}")
            return []

    def update(
        self, db: "Session", agent_id: int, **kwargs: Any
    ) -> Optional[Agent]:
        """更新智能体"""
        try:
            agent = self.get_by_id(db, agent_id)
            if not agent:
                return None
            for key, value in kwargs.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            db.commit()
            db.refresh(agent)
            return agent
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新智能体失败: {e}")
            return None

    def list_agents(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 50,
        status: Optional[AgentStatus] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Agent], int]:
        """列出智能体"""
        try:
            query = db.query(Agent)
            count_query = db.query(func.count(Agent.id))

            if status is not None:
                query = query.filter(Agent.status == status)
                count_query = count_query.filter(Agent.status == status)
            if category:
                query = query.filter(Agent.category == category)
                count_query = count_query.filter(Agent.category == category)
            if search:
                search_filter = or_(
                    Agent.name.ilike(f"%{search}%"),
                    Agent.description.ilike(f"%{search}%"),
                )
                query = query.filter(search_filter)
                count_query = count_query.filter(search_filter)

            total = count_query.scalar() or 0
            agents = query.order_by(desc(Agent.updated_at)).offset(skip).limit(limit).all()
            return agents, total
        except SQLAlchemyError as e:
            logger.error(f"列出智能体失败: {e}")
            return [], 0


# ============================================================
# 联盟 CRUD
# ============================================================

class AllianceCRUD(BaseCRUD):
    """
    智能体联盟 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(Alliance)

    def create(
        self,
        db: "Session",
        name: str,
        members: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[Alliance]:
        """创建联盟"""
        try:
            alliance = Alliance(
                name=name,
                members_json=members,
                current_member_count=len(members),
                **kwargs,
            )
            db.add(alliance)
            db.commit()
            db.refresh(alliance)
            logger.info(f"联盟创建成功: {name} (ID={alliance.id})")
            return alliance
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建联盟失败: {e}")
            return None

    def update(
        self, db: "Session", alliance_id: int, **kwargs: Any
    ) -> Optional[Alliance]:
        """更新联盟"""
        try:
            alliance = self.get_by_id(db, alliance_id)
            if not alliance:
                return None
            if "members_json" in kwargs and isinstance(kwargs["members_json"], list):
                alliance.current_member_count = len(kwargs["members_json"])
            for key, value in kwargs.items():
                if hasattr(alliance, key):
                    setattr(alliance, key, value)
            db.commit()
            db.refresh(alliance)
            return alliance
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新联盟失败: {e}")
            return None

    def get_active_alliances(self, db: "Session") -> List[Alliance]:
        """获取活跃的联盟"""
        try:
            return (
                db.query(Alliance)
                .filter(Alliance.is_active == True)  # noqa: E712
                .order_by(desc(Alliance.created_at))
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取活跃联盟失败: {e}")
            return []

    def add_member(
        self, db: "Session", alliance_id: int, member: Dict[str, Any]
    ) -> bool:
        """添加联盟成员"""
        try:
            alliance = self.get_by_id(db, alliance_id)
            if not alliance:
                return False
            members = alliance.members_json or []
            members.append(member)
            alliance.members_json = members
            alliance.current_member_count = len(members)
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"添加联盟成员失败: {e}")
            return False

    def remove_member(
        self, db: "Session", alliance_id: int, agent_id: int
    ) -> bool:
        """移除联盟成员"""
        try:
            alliance = self.get_by_id(db, alliance_id)
            if not alliance:
                return False
            members = alliance.members_json or []
            members = [m for m in members if m.get("agent_id") != agent_id]
            alliance.members_json = members
            alliance.current_member_count = len(members)
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"移除联盟成员失败: {e}")
            return False


# ============================================================
# 插件 CRUD
# ============================================================

class PluginCRUD(BaseCRUD):
    """
    插件 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(Plugin)

    def create(
        self,
        db: "Session",
        name: str,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> Optional[Plugin]:
        """创建插件"""
        try:
            plugin = Plugin(name=name, version=version, **kwargs)
            db.add(plugin)
            db.commit()
            db.refresh(plugin)
            logger.info(f"插件创建成功: {name} v{version} (ID={plugin.id})")
            return plugin
        except IntegrityError as e:
            db.rollback()
            logger.warning(f"插件创建失败（名称已存在）: {e}")
            return None
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建插件失败: {e}")
            return None

    def get_by_name(self, db: "Session", name: str) -> Optional[Plugin]:
        """根据名称获取插件"""
        try:
            return db.query(Plugin).filter(Plugin.name == name).first()
        except SQLAlchemyError as e:
            logger.error(f"根据名称获取插件失败: {e}")
            return None

    def update(
        self, db: "Session", plugin_id: int, **kwargs: Any
    ) -> Optional[Plugin]:
        """更新插件"""
        try:
            plugin = self.get_by_id(db, plugin_id)
            if not plugin:
                return None
            for key, value in kwargs.items():
                if hasattr(plugin, key):
                    setattr(plugin, key, value)
            db.commit()
            db.refresh(plugin)
            return plugin
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新插件失败: {e}")
            return None

    def set_status(
        self, db: "Session", plugin_id: int, status: PluginStatus
    ) -> bool:
        """设置插件状态"""
        return bool(self.update(db, plugin_id, status=status))

    def enable(self, db: "Session", plugin_id: int) -> bool:
        """启用插件"""
        return self.set_status(db, plugin_id, PluginStatus.ENABLED)

    def disable(self, db: "Session", plugin_id: int) -> bool:
        """禁用插件"""
        return self.set_status(db, plugin_id, PluginStatus.DISABLED)

    def list_plugins(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 100,
        status: Optional[PluginStatus] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Plugin], int]:
        """列出插件"""
        try:
            query = db.query(Plugin)
            count_query = db.query(func.count(Plugin.id))

            if status is not None:
                query = query.filter(Plugin.status == status)
                count_query = count_query.filter(Plugin.status == status)
            if category:
                query = query.filter(Plugin.category == category)
                count_query = count_query.filter(Plugin.category == category)
            if search:
                search_filter = or_(
                    Plugin.name.ilike(f"%{search}%"),
                    Plugin.display_name.ilike(f"%{search}%"),
                    Plugin.description.ilike(f"%{search}%"),
                )
                query = query.filter(search_filter)
                count_query = count_query.filter(search_filter)

            total = count_query.scalar() or 0
            plugins = query.order_by(asc(Plugin.name)).offset(skip).limit(limit).all()
            return plugins, total
        except SQLAlchemyError as e:
            logger.error(f"列出插件失败: {e}")
            return [], 0

    def update_usage(self, db: "Session", plugin_id: int) -> bool:
        """更新插件使用统计"""
        try:
            plugin = self.get_by_id(db, plugin_id)
            if plugin:
                plugin.total_uses = (plugin.total_uses or 0) + 1
                plugin.last_used_at = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新插件使用统计失败: {e}")
            return False


# ============================================================
# 渠道 CRUD
# ============================================================

class ChannelCRUD(BaseCRUD):
    """
    渠道 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(Channel)

    def create(
        self,
        db: "Session",
        name: str,
        channel_type: ChannelType,
        **kwargs: Any,
    ) -> Optional[Channel]:
        """创建渠道"""
        try:
            channel = Channel(name=name, channel_type=channel_type, **kwargs)
            db.add(channel)
            db.commit()
            db.refresh(channel)
            logger.info(f"渠道创建成功: {name} (类型={channel_type.value})")
            return channel
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建渠道失败: {e}")
            return None

    def get_by_type(
        self, db: "Session", channel_type: ChannelType
    ) -> List[Channel]:
        """按类型获取渠道"""
        try:
            return (
                db.query(Channel)
                .filter(Channel.channel_type == channel_type)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"按类型获取渠道失败: {e}")
            return []

    def get_active_channels(self, db: "Session") -> List[Channel]:
        """获取活跃渠道"""
        try:
            return (
                db.query(Channel)
                .filter(Channel.status == ChannelStatus.ACTIVE)
                .order_by(asc(Channel.name))
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取活跃渠道失败: {e}")
            return []

    def update(
        self, db: "Session", channel_id: int, **kwargs: Any
    ) -> Optional[Channel]:
        """更新渠道"""
        try:
            channel = self.get_by_id(db, channel_id)
            if not channel:
                return None
            for key, value in kwargs.items():
                if hasattr(channel, key):
                    setattr(channel, key, value)
            db.commit()
            db.refresh(channel)
            return channel
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新渠道失败: {e}")
            return None

    def set_status(
        self, db: "Session", channel_id: int, status: ChannelStatus
    ) -> bool:
        """设置渠道状态"""
        return bool(self.update(db, channel_id, status=status))

    def list_channels(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 100,
        channel_type: Optional[ChannelType] = None,
        status: Optional[ChannelStatus] = None,
    ) -> Tuple[List[Channel], int]:
        """列出渠道"""
        try:
            query = db.query(Channel)
            count_query = db.query(func.count(Channel.id))

            if channel_type is not None:
                query = query.filter(Channel.channel_type == channel_type)
                count_query = count_query.filter(Channel.channel_type == channel_type)
            if status is not None:
                query = query.filter(Channel.status == status)
                count_query = count_query.filter(Channel.status == status)

            total = count_query.scalar() or 0
            channels = query.order_by(asc(Channel.name)).offset(skip).limit(limit).all()
            return channels, total
        except SQLAlchemyError as e:
            logger.error(f"列出渠道失败: {e}")
            return [], 0


# ============================================================
# 数据集 CRUD
# ============================================================

class DatasetCRUD(BaseCRUD):
    """
    数据集 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(Dataset)

    def create(
        self,
        db: "Session",
        name: str,
        dataset_type: DatasetType = DatasetType.TEXT,
        **kwargs: Any,
    ) -> Optional[Dataset]:
        """创建数据集"""
        try:
            dataset = Dataset(name=name, type=dataset_type, **kwargs)
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
            logger.info(f"数据集创建成功: {name} (ID={dataset.id})")
            return dataset
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建数据集失败: {e}")
            return None

    def get_by_user(
        self,
        db: "Session",
        user_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[Dataset], int]:
        """获取用户的数据集"""
        try:
            query = db.query(Dataset).filter(Dataset.user_id == user_id)
            count_query = db.query(func.count(Dataset.id)).filter(
                Dataset.user_id == user_id
            )
            total = count_query.scalar() or 0
            datasets = (
                query.order_by(desc(Dataset.created_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
            return datasets, total
        except SQLAlchemyError as e:
            logger.error(f"获取用户数据集失败: {e}")
            return [], 0

    def get_public_datasets(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 50,
    ) -> List[Dataset]:
        """获取公开数据集"""
        try:
            return (
                db.query(Dataset)
                .filter(Dataset.is_public == True)  # noqa: E712
                .order_by(desc(Dataset.created_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取公开数据集失败: {e}")
            return []

    def update(
        self, db: "Session", dataset_id: int, **kwargs: Any
    ) -> Optional[Dataset]:
        """更新数据集"""
        try:
            dataset = self.get_by_id(db, dataset_id)
            if not dataset:
                return None
            for key, value in kwargs.items():
                if hasattr(dataset, key):
                    setattr(dataset, key, value)
            db.commit()
            db.refresh(dataset)
            return dataset
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新数据集失败: {e}")
            return None

    def list_datasets(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 50,
        dataset_type: Optional[DatasetType] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Dataset], int]:
        """列出数据集"""
        try:
            query = db.query(Dataset)
            count_query = db.query(func.count(Dataset.id))

            if dataset_type is not None:
                query = query.filter(Dataset.type == dataset_type)
                count_query = count_query.filter(Dataset.type == dataset_type)
            if category:
                query = query.filter(Dataset.category == category)
                count_query = count_query.filter(Dataset.category == category)
            if search:
                search_filter = or_(
                    Dataset.name.ilike(f"%{search}%"),
                    Dataset.description.ilike(f"%{search}%"),
                )
                query = query.filter(search_filter)
                count_query = count_query.filter(search_filter)

            total = count_query.scalar() or 0
            datasets = query.order_by(desc(Dataset.created_at)).offset(skip).limit(limit).all()
            return datasets, total
        except SQLAlchemyError as e:
            logger.error(f"列出数据集失败: {e}")
            return [], 0


# ============================================================
# 人格 CRUD
# ============================================================

class PersonalityCRUD(BaseCRUD):
    """
    人格 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(Personality)

    def create(
        self,
        db: "Session",
        name: str,
        user_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[Personality]:
        """创建人格"""
        try:
            personality = Personality(name=name, user_id=user_id, **kwargs)
            db.add(personality)
            db.commit()
            db.refresh(personality)
            logger.info(f"人格创建成功: {name} (ID={personality.id})")
            return personality
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建人格失败: {e}")
            return None

    def get_by_user(
        self,
        db: "Session",
        user_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[Personality], int]:
        """获取用户的人格"""
        try:
            query = db.query(Personality).filter(Personality.user_id == user_id)
            count_query = db.query(func.count(Personality.id)).filter(
                Personality.user_id == user_id
            )
            total = count_query.scalar() or 0
            personalities = (
                query.order_by(desc(Personality.updated_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
            return personalities, total
        except SQLAlchemyError as e:
            logger.error(f"获取用户人格失败: {e}")
            return [], 0

    def get_templates(self, db: "Session") -> List[Personality]:
        """获取人格模板"""
        try:
            return (
                db.query(Personality)
                .filter(Personality.is_template == True)  # noqa: E712
                .order_by(asc(Personality.name))
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取人格模板失败: {e}")
            return []

    def update(
        self, db: "Session", personality_id: int, **kwargs: Any
    ) -> Optional[Personality]:
        """更新人格"""
        try:
            personality = self.get_by_id(db, personality_id)
            if not personality:
                return None
            for key, value in kwargs.items():
                if hasattr(personality, key):
                    setattr(personality, key, value)
            personality.version = (personality.version or 0) + 1
            db.commit()
            db.refresh(personality)
            return personality
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新人格失败: {e}")
            return None

    def list_personalities(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 50,
        category: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Personality], int]:
        """列出人格"""
        try:
            query = db.query(Personality)
            count_query = db.query(func.count(Personality.id))

            if category:
                query = query.filter(Personality.category == category)
                count_query = count_query.filter(Personality.category == category)
            if search:
                search_filter = or_(
                    Personality.name.ilike(f"%{search}%"),
                    Personality.description.ilike(f"%{search}%"),
                )
                query = query.filter(search_filter)
                count_query = count_query.filter(search_filter)

            total = count_query.scalar() or 0
            personalities = query.order_by(desc(Personality.updated_at)).offset(skip).limit(limit).all()
            return personalities, total
        except SQLAlchemyError as e:
            logger.error(f"列出人格失败: {e}")
            return [], 0

    def create_version(
        self,
        db: "Session",
        personality_id: int,
        changelog: Optional[str] = None,
    ) -> Optional[Personality]:
        """创建人格新版本"""
        try:
            original = self.get_by_id(db, personality_id)
            if not original:
                return None
            new_personality = Personality(
                name=original.name,
                user_id=original.user_id,
                soul_md=original.soul_md,
                system_prompt=original.system_prompt,
                greeting=original.greeting,
                personality_traits=original.personality_traits,
                speaking_style=original.speaking_style,
                knowledge_areas=original.knowledge_areas,
                emotional_profile=original.emotional_profile,
                config_json=original.config_json,
                behavior_rules=original.behavior_rules,
                response_templates=original.response_templates,
                version=(original.version or 0) + 1,
                parent_id=original.id,
                changelog=changelog,
                is_public=original.is_public,
                is_template=original.is_template,
                category=original.category,
                tags=original.tags,
            )
            db.add(new_personality)
            db.commit()
            db.refresh(new_personality)
            return new_personality
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建人格版本失败: {e}")
            return None


# ============================================================
# 审计日志 CRUD
# ============================================================

class AuditLogCRUD(BaseCRUD):
    """
    审计日志 CRUD 操作类

    注意：审计日志通常只允许创建和查询，不允许修改和删除。
    """

    def __init__(self) -> None:
        super().__init__(AuditLog)

    def create(
        self,
        db: "Session",
        action: AuditAction,
        resource_type: ResourceType,
        user_id: Optional[int] = None,
        resource_id: Optional[int] = None,
        resource_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        status: str = "success",
        **kwargs: Any,
    ) -> Optional[AuditLog]:
        """创建审计日志"""
        try:
            log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_name=resource_name,
                details_json=details,
                ip_address=ip_address,
                status=status,
                **kwargs,
            )
            db.add(log)
            db.commit()
            db.refresh(log)
            return log
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建审计日志失败: {e}")
            return None

    def list_logs(
        self,
        db: "Session",
        skip: int = 0,
        limit: int = 100,
        user_id: Optional[int] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[ResourceType] = None,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[List[AuditLog], int]:
        """列出审计日志"""
        try:
            query = db.query(AuditLog)
            count_query = db.query(func.count(AuditLog.id))

            if user_id is not None:
                query = query.filter(AuditLog.user_id == user_id)
                count_query = count_query.filter(AuditLog.user_id == user_id)
            if action is not None:
                query = query.filter(AuditLog.action == action)
                count_query = count_query.filter(AuditLog.action == action)
            if resource_type is not None:
                query = query.filter(AuditLog.resource_type == resource_type)
                count_query = count_query.filter(AuditLog.resource_type == resource_type)
            if status is not None:
                query = query.filter(AuditLog.status == status)
                count_query = count_query.filter(AuditLog.status == status)
            if start_date is not None:
                query = query.filter(AuditLog.created_at >= start_date)
                count_query = count_query.filter(AuditLog.created_at >= start_date)
            if end_date is not None:
                query = query.filter(AuditLog.created_at <= end_date)
                count_query = count_query.filter(AuditLog.created_at <= end_date)

            total = count_query.scalar() or 0
            logs = query.order_by(desc(AuditLog.created_at)).offset(skip).limit(limit).all()
            return logs, total
        except SQLAlchemyError as e:
            logger.error(f"列出审计日志失败: {e}")
            return [], 0

    def search(
        self,
        db: "Session",
        search_term: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AuditLog]:
        """搜索审计日志"""
        try:
            query = db.query(AuditLog).filter(
                or_(
                    AuditLog.resource_name.ilike(f"%{search_term}%"),
                    AuditLog.ip_address.ilike(f"%{search_term}%"),
                    AuditLog.error_message.ilike(f"%{search_term}%"),
                    AuditLog.request_path.ilike(f"%{search_term}%"),
                )
            )
            return query.order_by(desc(AuditLog.created_at)).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"搜索审计日志失败: {e}")
            return []

    def get_user_activity(
        self,
        db: "Session",
        user_id: int,
        days: int = 30,
    ) -> Dict[str, Any]:
        """获取用户活动统计"""
        try:
            from datetime import timedelta
            start_date = datetime.now(timezone.utc) - timedelta(days=days)

            total = (
                db.query(func.count(AuditLog.id))
                .filter(
                    AuditLog.user_id == user_id,
                    AuditLog.created_at >= start_date,
                )
                .scalar() or 0
            )
            success = (
                db.query(func.count(AuditLog.id))
                .filter(
                    AuditLog.user_id == user_id,
                    AuditLog.created_at >= start_date,
                    AuditLog.status == "success",
                )
                .scalar() or 0
            )
            failed = (
                db.query(func.count(AuditLog.id))
                .filter(
                    AuditLog.user_id == user_id,
                    AuditLog.created_at >= start_date,
                    AuditLog.status == "error",
                )
                .scalar() or 0
            )
            return {
                "period_days": days,
                "total": total,
                "success": success,
                "failed": failed,
            }
        except SQLAlchemyError as e:
            logger.error(f"获取用户活动统计失败: {e}")
            return {"period_days": days, "total": 0, "success": 0, "failed": 0}

    def get_recent_errors(
        self,
        db: "Session",
        limit: int = 50,
    ) -> List[AuditLog]:
        """获取最近的错误日志"""
        try:
            return (
                db.query(AuditLog)
                .filter(AuditLog.status == "error")
                .order_by(desc(AuditLog.created_at))
                .limit(limit)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取最近错误日志失败: {e}")
            return []


# ============================================================
# 系统设置 CRUD
# ============================================================

class SystemSettingCRUD(BaseCRUD):
    """
    系统设置 CRUD 操作类
    """

    def __init__(self) -> None:
        super().__init__(SystemSetting)

    def get(self, db: "Session", key: str) -> Optional[SystemSetting]:
        """根据键名获取设置"""
        try:
            return db.query(SystemSetting).filter(SystemSetting.key == key).first()
        except SQLAlchemyError as e:
            logger.error(f"获取系统设置失败: {e}")
            return None

    def get_value(self, db: "Session", key: str, default: Any = None) -> Any:
        """获取设置值"""
        setting = self.get(db, key)
        if setting:
            return setting.get_typed_value()
        return default

    def set(
        self,
        db: "Session",
        key: str,
        value: Any,
        description: Optional[str] = None,
        value_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[SystemSetting]:
        """设置系统配置项"""
        try:
            setting = self.get(db, key)
            if setting:
                setting.value = str(value)
                if description is not None:
                    setting.description = description
                if value_type is not None:
                    setting.value_type = value_type
                for k, v in kwargs.items():
                    if hasattr(setting, k):
                        setattr(setting, k, v)
            else:
                if value_type is None:
                    if isinstance(value, bool):
                        value_type = "boolean"
                    elif isinstance(value, (int, float)):
                        value_type = "number"
                    elif isinstance(value, (dict, list)):
                        value_type = "json"
                    else:
                        value_type = "string"

                setting = SystemSetting(
                    key=key,
                    value=str(value),
                    description=description or "",
                    value_type=value_type,
                    **kwargs,
                )
                db.add(setting)

            db.commit()
            db.refresh(setting)
            return setting
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"设置系统配置失败: {e}")
            return None

    def get_by_category(
        self, db: "Session", category: str
    ) -> List[SystemSetting]:
        """按分类获取设置"""
        try:
            return (
                db.query(SystemSetting)
                .filter(SystemSetting.category == category)
                .order_by(asc(SystemSetting.key))
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"按分类获取设置失败: {e}")
            return []

    def get_public_settings(self, db: "Session") -> List[SystemSetting]:
        """获取公开的设置"""
        try:
            return (
                db.query(SystemSetting)
                .filter(SystemSetting.is_public == True)  # noqa: E712
                .order_by(asc(SystemSetting.key))
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取公开设置失败: {e}")
            return []

    def list_settings(
        self,
        db: "Session",
        category: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[SystemSetting]:
        """列出所有设置"""
        try:
            query = db.query(SystemSetting)
            if category:
                query = query.filter(SystemSetting.category == category)
            if search:
                query = query.filter(
                    or_(
                        SystemSetting.key.ilike(f"%{search}%"),
                        SystemSetting.description.ilike(f"%{search}%"),
                    )
                )
            return query.order_by(asc(SystemSetting.category), asc(SystemSetting.key)).all()
        except SQLAlchemyError as e:
            logger.error(f"列出设置失败: {e}")
            return []

    def delete(self, db: "Session", key: str) -> bool:
        """删除设置"""
        try:
            setting = self.get(db, key)
            if setting:
                db.delete(setting)
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"删除设置失败: {e}")
            return False

    def bulk_set(
        self,
        db: "Session",
        settings: Dict[str, Any],
        category: Optional[str] = None,
    ) -> int:
        """批量设置"""
        count = 0
        for key, value in settings.items():
            result = self.set(db, key, value, category=category)
            if result:
                count += 1
        return count
