"""
ORM Model Definitions 模块

提供 ORM 模型定义的基础组件：
- ORMBase: 声明式基类
- TimestampMixin: 时间戳混入
- SoftDeleteMixin: 软删除混入
- RelationshipHelper: 关系处理助手
- JSONType: JSON 类型装饰器
- EnumType: 枚举类型装饰器
- ValidationMixin: 验证混入

纯 Python 标准库实现，包含完整类型注解。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

T = TypeVar("T", bound="ORMBase")


# ============================================================
# 元数据和注册表
# ============================================================

class ModelRegistry:
    """模型注册表"""
    
    _models: Dict[str, Type[ORMBase]] = {}
    _lock = __import__("threading").Lock()
    
    @classmethod
    def register(cls, model_class: Type[ORMBase]) -> None:
        """注册模型"""
        with cls._lock:
            cls._models[model_class.__name__] = model_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[ORMBase]]:
        """获取模型"""
        return cls._models.get(name)
    
    @classmethod
    def all(cls) -> Dict[str, Type[ORMBase]]:
        """获取所有模型"""
        return cls._models.copy()
    
    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        with cls._lock:
            cls._models.clear()


# ============================================================
# 列定义
# ============================================================

class ColumnType(Enum):
    """列类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    TEXT = "text"
    BLOB = "blob"
    JSON = "json"
    UUID = "uuid"
    ENUM = "enum"


@dataclass
class Column:
    """列定义"""
    name: str
    type: ColumnType
    primary_key: bool = False
    nullable: bool = True
    default: Any = None
    unique: bool = False
    index: bool = False
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    auto_increment: bool = False
    comment: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.type.value,
            "primary_key": self.primary_key,
            "nullable": self.nullable,
            "default": self.default,
            "unique": self.unique,
            "index": self.index,
            "length": self.length,
            "precision": self.precision,
            "scale": self.scale,
            "auto_increment": self.auto_increment,
            "comment": self.comment,
        }


# ============================================================
# ORMBase - 声明式基类
# ============================================================

class ORMBase:
    """
    ORM 声明式基类
    
    所有 ORM 模型的基类，提供：
    - 自动表名生成
    - 列定义管理
    - 关系定义管理
    - 实例生命周期管理
    - 序列化/反序列化
    
    Attributes:
        __tablename__: 表名
        __columns__: 列定义字典
        __relationships__: 关系定义字典
        __indexes__: 索引定义列表
        __constraints__: 约束定义列表
    """
    
    __tablename__: Optional[str] = None
    __columns__: Dict[str, Column] = {}
    __relationships__: Dict[str, Dict[str, Any]] = {}
    __indexes__: List[Dict[str, Any]] = []
    __constraints__: List[Dict[str, Any]] = []
    
    def __init__(self, **kwargs: Any):
        """初始化实例"""
        self._data: Dict[str, Any] = {}
        self._dirty: Set[str] = set()
        self._new: bool = True
        self._deleted: bool = False
        self._loaded_relations: Set[str] = set()
        
        # 设置默认值
        for col_name, col in self.__class__.__columns__.items():
            if col.default is not None:
                self._data[col_name] = copy.deepcopy(col.default)
            else:
                self._data[col_name] = None
        
        # 应用传入的值
        for key, value in kwargs.items():
            if key in self.__class__.__columns__:
                self._set_value(key, value)
            elif key in self.__class__.__relationships__:
                setattr(self, f"_{key}", value)
                self._loaded_relations.add(key)
    
    def __init_subclass__(cls, **kwargs: Any):
        """子类初始化时注册"""
        super().__init_subclass__(**kwargs)
        
        # 设置默认表名
        if cls.__tablename__ is None:
            cls.__tablename__ = cls.__name__.lower() + "s"
        
        # 继承父类的列定义
        columns: Dict[str, Column] = {}
        relationships: Dict[str, Dict[str, Any]] = {}
        
        for base in cls.__mro__[1:]:
            if hasattr(base, "__columns__"):
                columns.update(getattr(base, "__columns__", {}))
            if hasattr(base, "__relationships__"):
                relationships.update(getattr(base, "__relationships__", {}))
        
        # 合并当前类的定义
        columns.update(getattr(cls, "__columns__", {}))
        relationships.update(getattr(cls, "__relationships__", {}))
        
        cls.__columns__ = columns
        cls.__relationships__ = relationships
        
        # 注册到全局注册表
        ModelRegistry.register(cls)
    
    def __getattr__(self, name: str) -> Any:
        """属性访问"""
        if name in self.__class__.__columns__:
            return self._data.get(name)
        if name in self.__class__.__relationships__:
            return getattr(self, f"_{name}", None)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """属性设置"""
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name in self.__class__.__columns__:
            self._set_value(name, value)
        else:
            super().__setattr__(name, value)
    
    def _set_value(self, name: str, value: Any) -> None:
        """设置字段值"""
        column = self.__class__.__columns__.get(name)
        if column:
            # 类型转换
            value = self._convert_type(value, column.type)
            self._data[name] = value
            self._dirty.add(name)
    
    def _convert_type(self, value: Any, col_type: ColumnType) -> Any:
        """类型转换"""
        if value is None:
            return None
        
        if col_type == ColumnType.STRING:
            return str(value)
        elif col_type == ColumnType.INTEGER:
            return int(value)
        elif col_type == ColumnType.FLOAT:
            return float(value)
        elif col_type == ColumnType.BOOLEAN:
            return bool(value)
        elif col_type == ColumnType.DATETIME:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value)
            elif isinstance(value, str):
                return datetime.fromisoformat(value)
            return value
        elif col_type == ColumnType.JSON:
            if isinstance(value, str):
                return json.loads(value)
            return value
        elif col_type == ColumnType.UUID:
            if isinstance(value, str):
                return uuid.UUID(value)
            return value
        
        return value
    
    @classmethod
    def _get_tablename(cls) -> str:
        """获取表名"""
        return cls.__tablename__ or cls.__name__.lower() + "s"
    
    @classmethod
    def _get_primary_key(cls) -> str:
        """获取主键列名"""
        for name, col in cls.__columns__.items():
            if col.primary_key:
                return name
        return "id"
    
    @classmethod
    def _get_columns(cls) -> Dict[str, Column]:
        """获取所有列"""
        return cls.__columns__
    
    @classmethod
    def _get_relationships(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有关系"""
        return cls.__relationships__
    
    def to_dict(self, exclude: Optional[Set[str]] = None, include_relations: bool = False) -> Dict[str, Any]:
        """
        转换为字典
        
        Args:
            exclude: 排除的字段
            include_relations: 是否包含关系字段
        """
        exclude = exclude or set()
        result: Dict[str, Any] = {}
        
        for name, value in self._data.items():
            if name in exclude:
                continue
            
            if hasattr(value, "to_dict"):
                result[name] = value.to_dict()
            elif isinstance(value, datetime):
                result[name] = value.isoformat()
            elif isinstance(value, uuid.UUID):
                result[name] = str(value)
            elif isinstance(value, Enum):
                result[name] = value.value
            else:
                result[name] = value
        
        if include_relations:
            for name in self._loaded_relations:
                value = getattr(self, f"_{name}", None)
                if value is not None:
                    if isinstance(value, list):
                        result[name] = [
                            item.to_dict() if hasattr(item, "to_dict") else item
                            for item in value
                        ]
                    elif hasattr(value, "to_dict"):
                        result[name] = value.to_dict()
                    else:
                        result[name] = value
        
        return result
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """从字典创建实例"""
        instance = cls.__new__(cls)
        ORMBase.__init__(instance)
        
        for key, value in data.items():
            if key in cls.__columns__:
                instance._data[key] = value
        
        instance._dirty.clear()
        instance._new = True
        return instance
    
    def clone(self: T) -> T:
        """克隆实例（新ID）"""
        data = self.to_dict()
        pk = self.__class__._get_primary_key()
        data.pop(pk, None)
        return self.__class__.from_dict(data)
    
    def is_dirty(self) -> bool:
        """检查是否有修改"""
        return len(self._dirty) > 0
    
    def get_dirty_fields(self) -> Set[str]:
        """获取修改的字段"""
        return self._dirty.copy()
    
    def mark_clean(self) -> None:
        """标记为干净状态"""
        self._dirty.clear()
        self._new = False
    
    def __repr__(self) -> str:
        pk = self.__class__._get_primary_key()
        pk_value = getattr(self, pk, None)
        return f"<{self.__class__.__name__} {pk}={pk_value!r}>"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ORMBase):
            return NotImplemented
        pk = self.__class__._get_primary_key()
        return getattr(self, pk, None) == getattr(other, pk, None)
    
    def __hash__(self) -> int:
        pk = self.__class__._get_primary_key()
        pk_value = getattr(self, pk, None)
        return hash(pk_value) if pk_value is not None else id(self)


# ============================================================
# TimestampMixin - 时间戳混入
# ============================================================

class TimestampMixin:
    """
    时间戳混入类
    
    自动管理 created_at 和 updated_at 字段。
    """
    
    __columns__ = {
        "created_at": Column(
            name="created_at",
            type=ColumnType.DATETIME,
            nullable=False,
            default=None,
        ),
        "updated_at": Column(
            name="updated_at",
            type=ColumnType.DATETIME,
            nullable=False,
            default=None,
        ),
    }
    
    def touch(self) -> None:
        """更新时间戳"""
        self.updated_at = datetime.now()  # type: ignore


# ============================================================
# SoftDeleteMixin - 软删除混入
# ============================================================

class SoftDeleteMixin:
    """
    软删除混入类
    
    提供软删除功能，通过 deleted_at 字段标记删除状态。
    """
    
    __columns__ = {
        "deleted_at": Column(
            name="deleted_at",
            type=ColumnType.DATETIME,
            nullable=True,
            default=None,
        ),
        "is_deleted": Column(
            name="is_deleted",
            type=ColumnType.BOOLEAN,
            nullable=False,
            default=False,
        ),
    }
    
    def soft_delete(self) -> None:
        """软删除"""
        self.deleted_at = datetime.now()  # type: ignore
        self.is_deleted = True  # type: ignore
    
    def restore(self) -> None:
        """恢复"""
        self.deleted_at = None  # type: ignore
        self.is_deleted = False  # type: ignore
    
    @property
    def is_deleted(self) -> bool:
        """是否已删除"""
        return getattr(self, "_is_deleted", False) or getattr(self, "deleted_at", None) is not None
    
    @is_deleted.setter
    def is_deleted(self, value: bool) -> None:
        """设置删除状态"""
        self._is_deleted = value


# ============================================================
# RelationshipHelper - 关系处理助手
# ============================================================

class RelationshipType(Enum):
    """关系类型"""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class RelationshipHelper:
    """
    关系处理助手
    
    提供声明式关系定义和查询功能。
    """
    
    @staticmethod
    def one_to_many(
        related_model: Type[ORMBase],
        foreign_key: str,
        back_populates: Optional[str] = None,
        lazy: bool = True,
    ) -> Dict[str, Any]:
        """
        定义一对多关系
        
        Args:
            related_model: 关联模型类
            foreign_key: 外键字段名
            back_populates: 反向关系字段名
            lazy: 是否延迟加载
        """
        return {
            "type": RelationshipType.ONE_TO_MANY,
            "related_model": related_model,
            "foreign_key": foreign_key,
            "back_populates": back_populates,
            "lazy": lazy,
        }
    
    @staticmethod
    def many_to_one(
        related_model: Type[ORMBase],
        foreign_key: str,
        back_populates: Optional[str] = None,
        lazy: bool = True,
    ) -> Dict[str, Any]:
        """
        定义多对一关系
        
        Args:
            related_model: 关联模型类
            foreign_key: 外键字段名
            back_populates: 反向关系字段名
            lazy: 是否延迟加载
        """
        return {
            "type": RelationshipType.MANY_TO_ONE,
            "related_model": related_model,
            "foreign_key": foreign_key,
            "back_populates": back_populates,
            "lazy": lazy,
        }
    
    @staticmethod
    def one_to_one(
        related_model: Type[ORMBase],
        foreign_key: str,
        back_populates: Optional[str] = None,
        lazy: bool = True,
    ) -> Dict[str, Any]:
        """
        定义一对一关系
        
        Args:
            related_model: 关联模型类
            foreign_key: 外键字段名
            back_populates: 反向关系字段名
            lazy: 是否延迟加载
        """
        return {
            "type": RelationshipType.ONE_TO_ONE,
            "related_model": related_model,
            "foreign_key": foreign_key,
            "back_populates": back_populates,
            "lazy": lazy,
        }
    
    @staticmethod
    def many_to_many(
        related_model: Type[ORMBase],
        join_table: str,
        local_key: str,
        remote_key: str,
        back_populates: Optional[str] = None,
        lazy: bool = True,
    ) -> Dict[str, Any]:
        """
        定义多对多关系
        
        Args:
            related_model: 关联模型类
            join_table: 关联表名
            local_key: 本地键字段名
            remote_key: 远程键字段名
            back_populates: 反向关系字段名
            lazy: 是否延迟加载
        """
        return {
            "type": RelationshipType.MANY_TO_MANY,
            "related_model": related_model,
            "join_table": join_table,
            "local_key": local_key,
            "remote_key": remote_key,
            "back_populates": back_populates,
            "lazy": lazy,
        }
    
    @staticmethod
    def eager_load(instance: ORMBase, *relation_names: str) -> ORMBase:
        """预加载关系"""
        for name in relation_names:
            if name in instance.__class__.__relationships__:
                instance._loaded_relations.add(name)
        return instance


# ============================================================
# JSONType - JSON 类型装饰器
# ============================================================

class JSONType:
    """
    JSON 类型装饰器
    
    自动处理 JSON 序列化和反序列化。
    """
    
    def __init__(
        self,
        default: Any = None,
        encoder: Optional[Type[json.JSONEncoder]] = None,
        decoder: Optional[Callable[[str], Any]] = None,
    ):
        self.default = default
        self.encoder = encoder
        self.decoder = decoder
    
    def to_database(self, value: Any) -> Optional[str]:
        """转换为数据库存储格式"""
        if value is None:
            return None
        if self.encoder:
            return json.dumps(value, cls=self.encoder)
        return json.dumps(value, default=self._default_serializer)
    
    def from_database(self, value: Optional[str]) -> Any:
        """从数据库格式转换"""
        if value is None:
            return self.default
        if self.decoder:
            return self.decoder(value)
        return json.loads(value)
    
    def _default_serializer(self, obj: Any) -> Any:
        """默认序列化器"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def create_column(self, name: str, nullable: bool = True) -> Column:
        """创建列定义"""
        return Column(
            name=name,
            type=ColumnType.JSON,
            nullable=nullable,
            default=self.default,
        )


# ============================================================
# EnumType - 枚举类型装饰器
# ============================================================

class EnumType:
    """
    枚举类型装饰器
    
    自动处理枚举值和数据库值的转换。
    """
    
    def __init__(
        self,
        enum_class: Type[Enum],
        by_value: bool = True,
        length: Optional[int] = None,
    ):
        self.enum_class = enum_class
        self.by_value = by_value
        self.length = length
    
    def to_database(self, value: Optional[Enum]) -> Any:
        """转换为数据库存储格式"""
        if value is None:
            return None
        return value.value if self.by_value else value.name
    
    def from_database(self, value: Any) -> Optional[Enum]:
        """从数据库格式转换"""
        if value is None:
            return None
        if self.by_value:
            return self.enum_class(value)
        return self.enum_class[value]
    
    def create_column(self, name: str, nullable: bool = True, default: Optional[Enum] = None) -> Column:
        """创建列定义"""
        default_value = default.value if default and self.by_value else default
        
        # 推断长度
        if self.length is None:
            if self.by_value:
                sample_values = [e.value for e in self.enum_class]
                self.length = max(len(str(v)) for v in sample_values) if sample_values else 50
            else:
                sample_names = [e.name for e in self.enum_class]
                self.length = max(len(n) for n in sample_names) if sample_names else 50
        
        return Column(
            name=name,
            type=ColumnType.ENUM,
            nullable=nullable,
            default=default_value,
            length=self.length,
        )


# ============================================================
# ValidationMixin - 验证混入
# ============================================================

class ValidationError(Exception):
    """验证错误"""
    
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error for '{field}': {message}")


class ValidationMixin:
    """
    验证混入类
    
    提供字段验证功能。
    """
    
    _validators: Dict[str, List[Callable[[Any], Optional[str]]]] = {}
    
    def validate(self) -> List[ValidationError]:
        """验证所有字段"""
        errors: List[ValidationError] = []
        
        for field_name, validators in self._get_validators().items():
            value = getattr(self, field_name, None)
            for validator in validators:
                error = validator(value)
                if error:
                    errors.append(ValidationError(field_name, error))
                    break
        
        return errors
    
    def is_valid(self) -> bool:
        """检查是否有效"""
        return len(self.validate()) == 0
    
    def validate_or_raise(self) -> None:
        """验证，失败则抛出异常"""
        errors = self.validate()
        if errors:
            raise errors[0]
    
    def _get_validators(self) -> Dict[str, List[Callable[[Any], Optional[str]]]]:
        """获取验证器字典"""
        return getattr(self, "_validators", {})
    
    @classmethod
    def add_validator(
        cls,
        field: str,
        validator: Callable[[Any], Optional[str]],
    ) -> None:
        """添加验证器"""
        if field not in cls._validators:
            cls._validators[field] = []
        cls._validators[field].append(validator)


# ============================================================
# 验证器工厂
# ============================================================

class Validators:
    """验证器工厂类"""
    
    @staticmethod
    def required(message: str = "Field is required") -> Callable[[Any], Optional[str]]:
        """必填验证"""
        def validator(value: Any) -> Optional[str]:
            if value is None or (isinstance(value, str) and not value.strip()):
                return message
            return None
        return validator
    
    @staticmethod
    def min_length(min_len: int, message: Optional[str] = None) -> Callable[[Any], Optional[str]]:
        """最小长度验证"""
        msg = message or f"Minimum length is {min_len}"
        def validator(value: Any) -> Optional[str]:
            if value is not None and len(str(value)) < min_len:
                return msg
            return None
        return validator
    
    @staticmethod
    def max_length(max_len: int, message: Optional[str] = None) -> Callable[[Any], Optional[str]]:
        """最大长度验证"""
        msg = message or f"Maximum length is {max_len}"
        def validator(value: Any) -> Optional[str]:
            if value is not None and len(str(value)) > max_len:
                return msg
            return None
        return validator
    
    @staticmethod
    def pattern(regex: str, message: Optional[str] = None) -> Callable[[Any], Optional[str]]:
        """正则表达式验证"""
        msg = message or f"Value does not match pattern"
        compiled = re.compile(regex)
        def validator(value: Any) -> Optional[str]:
            if value is not None and not compiled.match(str(value)):
                return msg
            return None
        return validator
    
    @staticmethod
    def email(message: str = "Invalid email format") -> Callable[[Any], Optional[str]]:
        """邮箱格式验证"""
        return Validators.pattern(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            message
        )
    
    @staticmethod
    def range(
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        message: Optional[str] = None,
    ) -> Callable[[Any], Optional[str]]:
        """范围验证"""
        def validator(value: Any) -> Optional[str]:
            if value is None:
                return None
            try:
                num = float(value)
                if min_val is not None and num < min_val:
                    return message or f"Value must be >= {min_val}"
                if max_val is not None and num > max_val:
                    return message or f"Value must be <= {max_val}"
            except (ValueError, TypeError):
                return "Value must be a number"
            return None
        return validator
    
    @staticmethod
    def one_of(values: List[Any], message: Optional[str] = None) -> Callable[[Any], Optional[str]]:
        """枚举值验证"""
        msg = message or f"Value must be one of {values}"
        def validator(value: Any) -> Optional[str]:
            if value is not None and value not in values:
                return msg
            return None
        return validator
    
    @staticmethod
    def custom(check: Callable[[Any], bool], message: str) -> Callable[[Any], Optional[str]]:
        """自定义验证"""
        def validator(value: Any) -> Optional[str]:
            if value is not None and not check(value):
                return message
            return None
        return validator


# ============================================================
# 辅助函数
# ============================================================

def declarative_base() -> Type[ORMBase]:
    """创建声明式基类"""
    return ORMBase


def ColumnDef(
    type_: ColumnType,
    primary_key: bool = False,
    nullable: bool = True,
    default: Any = None,
    unique: bool = False,
    index: bool = False,
    length: Optional[int] = None,
    **kwargs: Any,
) -> Column:
    """创建列定义"""
    return Column(
        name="",  # 将在模型定义时设置
        type=type_,
        primary_key=primary_key,
        nullable=nullable,
        default=default,
        unique=unique,
        index=index,
        length=length,
        **kwargs,
    )


def relationship(
    related_model: Type[ORMBase],
    relation_type: RelationshipType = RelationshipType.MANY_TO_ONE,
    **kwargs: Any,
) -> Dict[str, Any]:
    """创建关系定义"""
    if relation_type == RelationshipType.ONE_TO_MANY:
        return RelationshipHelper.one_to_many(related_model, **kwargs)
    elif relation_type == RelationshipType.MANY_TO_ONE:
        return RelationshipHelper.many_to_one(related_model, **kwargs)
    elif relation_type == RelationshipType.ONE_TO_ONE:
        return RelationshipHelper.one_to_one(related_model, **kwargs)
    elif relation_type == RelationshipType.MANY_TO_MANY:
        return RelationshipHelper.many_to_many(related_model, **kwargs)
    raise ValueError(f"Unknown relationship type: {relation_type}")
