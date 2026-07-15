"""
存储抽象层基础模块

定义存储接口、实体基类、查询过滤器、排序、分页和查询结果等核心抽象。
"""

import copy
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

# 泛型类型变量
T = TypeVar("T", bound="Entity")


# ============================================================
# 实体基类
# ============================================================

class Entity:
    """
    实体基类

    所有存储实体的基类，提供统一的ID和时间戳管理。
    支持通过字典初始化和导出。

    Attributes:
        id: 实体唯一标识符（UUID字符串）
        created_at: 创建时间戳
        updated_at: 更新时间戳
        metadata: 自定义元数据字典
    """

    def __init__(
        self,
        id: Optional[str] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.id = id or str(uuid.uuid4())
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or self.created_at
        self.metadata = metadata or {}
        # 将额外的关键字参数作为属性存储
        for key, value in kwargs.items():
            if not key.startswith("_"):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """将实体转换为字典"""
        result: Dict[str, Any] = {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": copy.deepcopy(self.metadata),
        }
        # 收集所有非内部属性
        for key, value in self.__dict__.items():
            if key not in ("id", "created_at", "updated_at", "metadata") and not key.startswith("_"):
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    result[key] = copy.deepcopy(value) if isinstance(value, (list, dict)) else value
                else:
                    result[key] = str(value)
        return result

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """从字典创建实体"""
        data = copy.deepcopy(data)
        entity_id = data.pop("id", None)
        created_at = data.pop("created_at", None)
        updated_at = data.pop("updated_at", None)
        metadata = data.pop("metadata", None)
        return cls(
            id=entity_id,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            **data,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"<{cls_name} id={self.id!r}>"

    def clone(self: Type[T]) -> T:
        """创建实体的深拷贝（新ID和新时间戳）"""
        data = self.to_dict()
        data.pop("id")
        data.pop("created_at")
        data.pop("updated_at")
        return self.__class__.from_dict(data)

    def touch(self) -> None:
        """更新 updated_at 时间戳"""
        self.updated_at = time.time()


# ============================================================
# 查询过滤器
# ============================================================

class FilterOperator(Enum):
    """过滤操作符枚举"""
    EQ = "eq"           # 等于
    NE = "ne"           # 不等于
    GT = "gt"           # 大于
    GTE = "gte"         # 大于等于
    LT = "lt"           # 小于
    LTE = "lte"         # 小于等于
    IN = "in"           # 包含于
    NOT_IN = "not_in"   # 不包含于
    CONTAINS = "contains"       # 字符串包含
    STARTSWITH = "startswith"   # 字符串前缀
    ENDSWITH = "endswith"       # 字符串后缀
    REGEX = "regex"             # 正则匹配
    IS_NULL = "is_null"         # 为空
    IS_NOT_NULL = "is_not_null" # 不为空
    BETWEEN = "between"         # 区间


@dataclass
class QueryFilter:
    """
    查询过滤器

    定义单个查询条件，支持多种比较操作符。

    Attributes:
        field: 要过滤的字段名
        operator: 过滤操作符
        value: 比较值
        logic: 逻辑连接符（and/or），用于组合多个过滤器
    """
    field: str
    operator: FilterOperator = FilterOperator.EQ
    value: Any = None
    logic: str = "and"

    def matches(self, entity: Entity) -> bool:
        """
        检查实体是否匹配此过滤器

        Args:
            entity: 待检查的实体

        Returns:
            是否匹配
        """
        field_value = self._get_field_value(entity)

        if self.operator == FilterOperator.EQ:
            return field_value == self.value
        elif self.operator == FilterOperator.NE:
            return field_value != self.value
        elif self.operator == FilterOperator.GT:
            if field_value is None:
                return False
            return field_value > self.value
        elif self.operator == FilterOperator.GTE:
            if field_value is None:
                return False
            return field_value >= self.value
        elif self.operator == FilterOperator.LT:
            if field_value is None:
                return False
            return field_value < self.value
        elif self.operator == FilterOperator.LTE:
            if field_value is None:
                return False
            return field_value <= self.value
        elif self.operator == FilterOperator.IN:
            return field_value in self.value
        elif self.operator == FilterOperator.NOT_IN:
            return field_value not in self.value
        elif self.operator == FilterOperator.CONTAINS:
            if field_value is None or self.value is None:
                return False
            return self.value in str(field_value)
        elif self.operator == FilterOperator.STARTSWITH:
            if field_value is None or self.value is None:
                return False
            return str(field_value).startswith(str(self.value))
        elif self.operator == FilterOperator.ENDSWITH:
            if field_value is None or self.value is None:
                return False
            return str(field_value).endswith(str(self.value))
        elif self.operator == FilterOperator.REGEX:
            import re
            if field_value is None:
                return False
            return bool(re.search(self.value, str(field_value)))
        elif self.operator == FilterOperator.IS_NULL:
            return field_value is None
        elif self.operator == FilterOperator.IS_NOT_NULL:
            return field_value is not None
        elif self.operator == FilterOperator.BETWEEN:
            if field_value is None:
                return False
            low, high = self.value
            return low <= field_value <= high
        return False

    def _get_field_value(self, entity: Entity) -> Any:
        """获取实体上指定字段的值，支持嵌套点号路径"""
        obj = entity
        parts = self.field.split(".")
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "logic": self.logic,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryFilter":
        """从字典创建"""
        data = copy.deepcopy(data)
        op_str = data.pop("operator", "eq")
        operator = FilterOperator(op_str) if isinstance(op_str, str) else op_str
        return cls(operator=operator, **data)


# ============================================================
# 排序
# ============================================================

class SortDirection(Enum):
    """排序方向"""
    ASC = "asc"   # 升序
    DESC = "desc"  # 降序


@dataclass
class SortOrder:
    """
    排序规则

    Attributes:
        field: 排序字段名
        direction: 排序方向（升序/降序）
    """
    field: str
    direction: SortDirection = SortDirection.ASC

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "field": self.field,
            "direction": self.direction.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SortOrder":
        """从字典创建"""
        data = copy.deepcopy(data)
        dir_str = data.pop("direction", "asc")
        direction = SortDirection(dir_str) if isinstance(dir_str, str) else dir_str
        return cls(direction=direction, **data)


# ============================================================
# 分页
# ============================================================

@dataclass
class Pagination:
    """
    分页参数

    Attributes:
        page: 页码（从1开始）
        page_size: 每页数量
        offset: 偏移量（与page二选一，优先使用offset）
        limit: 限制数量（与page_size二选一，优先使用limit）
    """
    page: int = 1
    page_size: int = 20
    offset: Optional[int] = None
    limit: Optional[int] = None

    @property
    def effective_offset(self) -> int:
        """计算有效的偏移量"""
        if self.offset is not None:
            return self.offset
        return (self.page - 1) * self.page_size

    @property
    def effective_limit(self) -> int:
        """计算有效的限制数量"""
        if self.limit is not None:
            return self.limit
        return self.page_size

    @property
    def total_pages(self, total_count: int) -> int:
        """计算总页数"""
        limit = self.effective_limit
        if limit <= 0:
            return 0
        return max(1, (total_count + limit - 1) // limit)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "page": self.page,
            "page_size": self.page_size,
            "offset": self.offset,
            "limit": self.limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pagination":
        """从字典创建"""
        return cls(**data)


# ============================================================
# 查询结果
# ============================================================

@dataclass
class QueryResult:
    """
    查询结果

    封装分页查询的结果集和元数据。

    Attributes:
        items: 结果项列表
        total: 符合条件的总记录数
        page: 当前页码
        page_size: 每页数量
    """
    items: List[Any] = dataclass_field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20

    @property
    def has_next(self) -> bool:
        """是否有下一页"""
        return self.page * self.page_size < self.total

    @property
    def has_prev(self) -> bool:
        """是否有上一页"""
        return self.page > 1

    @property
    def total_pages(self) -> int:
        """总页数"""
        if self.page_size <= 0:
            return 0
        return max(1, (self.total + self.page_size - 1) // self.page_size)

    @property
    def is_empty(self) -> bool:
        """结果是否为空"""
        return len(self.items) == 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        items_data = []
        for item in self.items:
            if hasattr(item, "to_dict"):
                items_data.append(item.to_dict())
            elif isinstance(item, dict):
                items_data.append(item)
            else:
                items_data.append(str(item))
        return {
            "items": items_data,
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
            "total_pages": self.total_pages,
        }


# ============================================================
# 存储抽象接口
# ============================================================

class Repository(ABC, Generic[T]):
    """
    存储抽象接口

    定义统一的CRUD操作接口，所有存储实现都必须继承此抽象类。
    支持泛型，可以指定具体的实体类型。

    Type Parameters:
        T: 实体类型，必须继承自Entity
    """

    @abstractmethod
    def create(self, entity: T) -> T:
        """
        创建实体

        Args:
            entity: 要创建的实体

        Returns:
            创建后的实体（可能包含自动生成的ID和时间戳）
        """
        ...

    @abstractmethod
    def read(self, id: str) -> Optional[T]:
        """
        读取实体

        Args:
            id: 实体ID

        Returns:
            实体对象，不存在则返回None
        """
        ...

    @abstractmethod
    def update(self, id: str, data: Dict[str, Any]) -> Optional[T]:
        """
        更新实体

        Args:
            id: 实体ID
            data: 要更新的字段和值

        Returns:
            更新后的实体，不存在则返回None
        """
        ...

    @abstractmethod
    def delete(self, id: str) -> bool:
        """
        删除实体

        Args:
            id: 实体ID

        Returns:
            是否删除成功
        """
        ...

    @abstractmethod
    def query(
        self,
        filters: Optional[List[QueryFilter]] = None,
        sort: Optional[List[SortOrder]] = None,
        pagination: Optional[Pagination] = None,
    ) -> QueryResult:
        """
        查询实体

        Args:
            filters: 过滤条件列表
            sort: 排序规则列表
            pagination: 分页参数

        Returns:
            查询结果（包含分页信息）
        """
        ...

    @abstractmethod
    def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """
        计数

        Args:
            filters: 过滤条件列表

        Returns:
            符合条件的记录数
        """
        ...

    @abstractmethod
    def exists(self, id: str) -> bool:
        """
        检查实体是否存在

        Args:
            id: 实体ID

        Returns:
            是否存在
        """
        ...

    def find_one(
        self,
        filters: Optional[List[QueryFilter]] = None,
        sort: Optional[List[SortOrder]] = None,
    ) -> Optional[T]:
        """
        查找单个实体

        默认实现基于query()，子类可覆盖以优化性能。

        Args:
            filters: 过滤条件列表
            sort: 排序规则列表

        Returns:
            第一个匹配的实体，无匹配则返回None
        """
        result = self.query(filters=filters, sort=sort, pagination=Pagination(page=1, page_size=1))
        if result.items:
            return result.items[0]
        return None

    def find_by_id(self, id: str) -> Optional[T]:
        """read的别名"""
        return self.read(id)

    def get_all(
        self,
        sort: Optional[List[SortOrder]] = None,
    ) -> List[T]:
        """
        获取所有实体

        Args:
            sort: 排序规则列表

        Returns:
            所有实体列表
        """
        result = self.query(filters=None, sort=sort, pagination=None)
        return result.items

    def create_or_update(self, entity: T) -> T:
        """
        创建或更新实体

        如果实体ID已存在则更新，否则创建。

        Args:
            entity: 实体对象

        Returns:
            创建或更新后的实体
        """
        existing = self.read(entity.id)
        if existing is not None:
            data = entity.to_dict()
            data.pop("id", None)
            data.pop("created_at", None)
            return self.update(entity.id, data)  # type: ignore
        return self.create(entity)

    def delete_where(self, filters: List[QueryFilter]) -> int:
        """
        按条件删除

        Args:
            filters: 过滤条件列表

        Returns:
            删除的记录数
        """
        result = self.query(filters=filters)
        count = 0
        for item in result.items:
            if self.delete(item.id):
                count += 1
        return count
