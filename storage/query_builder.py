"""
动态查询构建器

提供链式API构建查询条件，支持条件组合、排序和分页。
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import (
    FilterOperator,
    Pagination,
    QueryFilter,
    QueryResult,
    SortDirection,
    SortOrder,
)


class QueryBuilderError(Exception):
    """查询构建器异常"""
    pass


@dataclass
class WhereClause:
    """
    WHERE子句

    支持AND/OR逻辑组合。
    """
    filters: List[QueryFilter] = field(default_factory=list)
    logic: str = "and"  # "and" or "or"
    nested: List["WhereClause"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result: Dict[str, Any] = {
            "logic": self.logic,
            "filters": [f.to_dict() for f in self.filters],
        }
        if self.nested:
            result["nested"] = [n.to_dict() for n in self.nested]
        return result


@dataclass
class BuiltQuery:
    """
    构建完成的查询对象

    Attributes:
        filters: 过滤条件列表
        sort: 排序规则列表
        pagination: 分页参数
        select_fields: 选择字段列表
        where_clauses: WHERE子句树
    """
    filters: List[QueryFilter] = field(default_factory=list)
    sort: List[SortOrder] = field(default_factory=list)
    pagination: Optional[Pagination] = None
    select_fields: Optional[List[str]] = None
    where_clauses: List[WhereClause] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result: Dict[str, Any] = {
            "filters": [f.to_dict() for f in self.filters],
            "sort": [s.to_dict() for s in self.sort],
        }
        if self.pagination:
            result["pagination"] = self.pagination.to_dict()
        if self.select_fields:
            result["select_fields"] = self.select_fields
        if self.where_clauses:
            result["where_clauses"] = [w.to_dict() for w in self.where_clauses]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuiltQuery":
        """从字典创建"""
        filters = [
            QueryFilter.from_dict(f) for f in data.get("filters", [])
        ]
        sort = [
            SortOrder.from_dict(s) for s in data.get("sort", [])
        ]
        pagination = None
        if "pagination" in data:
            pagination = Pagination.from_dict(data["pagination"])
        return cls(
            filters=filters,
            sort=sort,
            pagination=pagination,
            select_fields=data.get("select_fields"),
        )


class QueryBuilder:
    """
    动态查询构建器

    提供流畅的链式API来构建查询条件。

    Usage:
        query = (QueryBuilder()
            .select("name", "email", "age")
            .where("age", FilterOperator.GTE, 18)
            .and_("status", FilterOperator.EQ, "active")
            .order_by("created_at", SortDirection.DESC)
            .limit(10)
            .offset(0)
            .build())

    支持的功能：
        - select(): 选择字段
        - where(): 添加条件
        - and_(): AND组合条件
        - or_(): OR组合条件
        - order_by(): 排序
        - limit(): 限制数量
        - offset(): 偏移量
        - page()/page_size(): 分页
        - build(): 构建查询对象
        - reset(): 重置构建器
        - clone(): 克隆构建器
    """

    def __init__(self):
        self._filters: List[QueryFilter] = []
        self._sort_orders: List[SortOrder] = []
        self._pagination: Optional[Pagination] = None
        self._select_fields: Optional[List[str]] = None
        self._where_clauses: List[WhereClause] = []
        self._current_logic = "and"

    def select(self, *fields: str) -> "QueryBuilder":
        """
        选择字段

        Args:
            fields: 字段名列表

        Returns:
            self（支持链式调用）
        """
        self._select_fields = list(fields)
        return self

    def where(
        self,
        field: str,
        operator: Union[FilterOperator, str] = FilterOperator.EQ,
        value: Any = None,
    ) -> "QueryBuilder":
        """
        添加WHERE条件

        Args:
            field: 字段名
            operator: 操作符（FilterOperator枚举或字符串）
            value: 比较值

        Returns:
            self
        """
        if isinstance(operator, str):
            operator = FilterOperator(operator)

        f = QueryFilter(
            field=field,
            operator=operator,
            value=value,
            logic=self._current_logic,
        )
        self._filters.append(f)
        return self

    def and_(self) -> "QueryBuilder":
        """
        切换到AND逻辑

        Returns:
            self
        """
        self._current_logic = "and"
        return self

    def or_(self) -> "QueryBuilder":
        """
        切换到OR逻辑

        Returns:
            self
        """
        self._current_logic = "or"
        return self

    def order_by(
        self,
        field: str,
        direction: Union[SortDirection, str] = SortDirection.ASC,
    ) -> "QueryBuilder":
        """
        添加排序规则

        Args:
            field: 排序字段
            direction: 排序方向

        Returns:
            self
        """
        if isinstance(direction, str):
            direction = SortDirection(direction)

        order = SortOrder(field=field, direction=direction)
        self._sort_orders.append(order)
        return self

    def asc(self, field: str) -> "QueryBuilder":
        """升序排序的快捷方法"""
        return self.order_by(field, SortDirection.ASC)

    def desc(self, field: str) -> "QueryBuilder":
        """降序排序的快捷方法"""
        return self.order_by(field, SortDirection.DESC)

    def limit(self, count: int) -> "QueryBuilder":
        """
        限制结果数量

        Args:
            count: 最大返回数量

        Returns:
            self
        """
        if self._pagination is None:
            self._pagination = Pagination()
        self._pagination.limit = count
        return self

    def offset(self, skip: int) -> "QueryBuilder":
        """
        设置偏移量

        Args:
            skip: 跳过的记录数

        Returns:
            self
        """
        if self._pagination is None:
            self._pagination = Pagination()
        self._pagination.offset = skip
        return self

    def page(self, page_num: int) -> "QueryBuilder":
        """
        设置页码

        Args:
            page_num: 页码（从1开始）

        Returns:
            self
        """
        if self._pagination is None:
            self._pagination = Pagination()
        self._pagination.page = max(1, page_num)
        return self

    def page_size(self, size: int) -> "QueryBuilder":
        """
        设置每页数量

        Args:
            size: 每页记录数

        Returns:
            self
        """
        if self._pagination is None:
            self._pagination = Pagination()
        self._pagination.page_size = max(1, size)
        return self

    def between(self, field: str, low: Any, high: Any) -> "QueryBuilder":
        """
        区间查询的快捷方法

        Args:
            field: 字段名
            low: 下界
            high: 上界

        Returns:
            self
        """
        return self.where(field, FilterOperator.BETWEEN, (low, high))

    def in_(self, field: str, values: List[Any]) -> "QueryBuilder":
        """
        IN查询的快捷方法

        Args:
            field: 字段名
            values: 值列表

        Returns:
            self
        """
        return self.where(field, FilterOperator.IN, values)

    def not_in(self, field: str, values: List[Any]) -> "QueryBuilder":
        """
        NOT IN查询的快捷方法

        Args:
            field: 字段名
            values: 值列表

        Returns:
            self
        """
        return self.where(field, FilterOperator.NOT_IN, values)

    def like(self, field: str, pattern: str) -> "QueryBuilder":
        """
        模糊匹配的快捷方法（使用contains）

        Args:
            field: 字段名
            pattern: 匹配模式

        Returns:
            self
        """
        return self.where(field, FilterOperator.CONTAINS, pattern)

    def is_null(self, field: str) -> "QueryBuilder":
        """
        IS NULL的快捷方法

        Args:
            field: 字段名

        Returns:
            self
        """
        return self.where(field, FilterOperator.IS_NULL, None)

    def is_not_null(self, field: str) -> "QueryBuilder":
        """
        IS NOT NULL的快捷方法

        Args:
            field: 字段名

        Returns:
            self
        """
        return self.where(field, FilterOperator.IS_NOT_NULL, None)

    def regex(self, field: str, pattern: str) -> "QueryBuilder":
        """
        正则匹配的快捷方法

        Args:
            field: 字段名
            pattern: 正则表达式

        Returns:
            self
        """
        return self.where(field, FilterOperator.REGEX, pattern)

    def group(
        self,
        builder_fn: Optional[callable] = None,
        logic: str = "and",
    ) -> "QueryBuilder":
        """
        创建条件分组

        Args:
            builder_fn: 构建函数，接收QueryBuilder参数
            logic: 组内逻辑（and/or）

        Returns:
            self
        """
        if builder_fn is not None:
            sub_builder = QueryBuilder()
            builder_fn(sub_builder)
            clause = WhereClause(
                filters=list(sub_builder._filters),
                logic=logic,
            )
            self._where_clauses.append(clause)
        return self

    def build(self) -> BuiltQuery:
        """
        构建查询对象

        Returns:
            构建完成的查询对象
        """
        # 合并where_clauses中的过滤器到主过滤器列表
        all_filters = list(self._filters)
        for clause in self._where_clauses:
            for f in clause.filters:
                f_copy = copy.deepcopy(f)
                f_copy.logic = clause.logic
                all_filters.append(f_copy)

        return BuiltQuery(
            filters=all_filters,
            sort=list(self._sort_orders),
            pagination=copy.deepcopy(self._pagination),
            select_fields=list(self._select_fields) if self._select_fields else None,
            where_clauses=list(self._where_clauses),
        )

    def reset(self) -> "QueryBuilder":
        """
        重置构建器

        Returns:
            self
        """
        self._filters = []
        self._sort_orders = []
        self._pagination = None
        self._select_fields = None
        self._where_clauses = []
        self._current_logic = "and"
        return self

    def clone(self) -> "QueryBuilder":
        """
        克隆构建器

        Returns:
            新的QueryBuilder实例
        """
        new_builder = QueryBuilder()
        new_builder._filters = copy.deepcopy(self._filters)
        new_builder._sort_orders = copy.deepcopy(self._sort_orders)
        new_builder._pagination = copy.deepcopy(self._pagination)
        new_builder._select_fields = list(self._select_fields) if self._select_fields else None
        new_builder._where_clauses = copy.deepcopy(self._where_clauses)
        new_builder._current_logic = self._current_logic
        return new_builder

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.build().to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryBuilder":
        """从字典创建"""
        built = BuiltQuery.from_dict(data)
        builder = cls()
        builder._filters = built.filters
        builder._sort_orders = built.sort
        builder._pagination = built.pagination
        builder._select_fields = built.select_fields
        builder._where_clauses = built.where_clauses
        return builder

    def __repr__(self) -> str:
        parts = []
        if self._select_fields:
            parts.append(f"select({', '.join(self._select_fields)})")
        if self._filters:
            parts.append(f"filters({len(self._filters)})")
        if self._sort_orders:
            parts.append(f"sort({len(self._sort_orders)})")
        if self._pagination:
            parts.append(f"page({self._pagination.page})")
        return f"QueryBuilder<{', '.join(parts)}>"
