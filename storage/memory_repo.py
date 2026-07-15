"""
内存存储实现

基于字典的内存存储，支持完整的CRUD操作、过滤、排序、分页、
批量操作和事务支持。
"""

import copy
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from .base import (
    Entity,
    FilterOperator,
    Pagination,
    QueryFilter,
    QueryResult,
    Repository,
    SortDirection,
    SortOrder,
)

T = TypeVar("T", bound="Entity")


class TransactionError(Exception):
    """事务操作异常"""
    pass


class Transaction:
    """
    事务上下文

    跟踪事务中的所有操作，支持提交和回滚。
    """

    def __init__(self, repo: "MemoryRepository"):
        self._repo = repo
        self._original_data: Optional[Dict[str, Dict[str, Any]]] = None
        self._operations: List[Dict[str, Any]] = []
        self._active = False
        self._lock = repo._lock

    def begin(self) -> None:
        """开始事务"""
        if self._active:
            raise TransactionError("事务已经处于活跃状态")
        with self._lock:
            self._original_data = copy.deepcopy(self._repo._store)
            self._operations = []
            self._active = True

    def commit(self) -> None:
        """提交事务"""
        if not self._active:
            raise TransactionError("没有活跃的事务")
        with self._lock:
            self._active = False
            self._original_data = None
            self._operations = []

    def rollback(self) -> None:
        """回滚事务"""
        if not self._active:
            raise TransactionError("没有活跃的事务")
        with self._lock:
            self._repo._store = self._original_data or {}
            self._active = False
            self._original_data = None
            self._operations = []

    @property
    def is_active(self) -> bool:
        """事务是否活跃"""
        return self._active

    def record_operation(self, op_type: str, entity_id: str, data: Any = None) -> None:
        """记录操作"""
        self._operations.append({
            "type": op_type,
            "id": entity_id,
            "data": data,
            "timestamp": time.time(),
        })


class MemoryRepository(Repository[T]):
    """
    内存存储实现

    基于Python字典的线程安全内存存储，提供完整的CRUD操作。
    支持丰富的过滤操作符、排序、分页、批量操作和事务。

    Features:
        - 线程安全（threading.Lock）
        - 自动UUID生成
        - 自动时间戳管理
        - 丰富的过滤操作符（eq/ne/gt/gte/lt/lte/in/contains/regex等）
        - 多字段排序
        - 分页查询
        - 批量操作（bulk_create/bulk_update/bulk_delete）
        - 事务支持（begin/commit/rollback）
    """

    def __init__(self, entity_class: Optional[Type[T]] = None):
        """
        初始化内存存储

        Args:
            entity_class: 实体类（用于反序列化），为None时使用Entity
        """
        self._store: Dict[str, Dict[str, Any]] = {}
        self._entity_class = entity_class or Entity
        self._lock = threading.RLock()
        self._transaction = Transaction(self)
        self._id_counter = 0

    # ============================================================
    # CRUD 操作
    # ============================================================

    def create(self, entity: T) -> T:
        """创建实体"""
        with self._lock:
            if entity.id is None:
                entity.id = self._generate_id()

            now = time.time()
            if entity.created_at is None:
                entity.created_at = now
            if entity.updated_at is None:
                entity.updated_at = now

            data = entity.to_dict()
            self._store[entity.id] = data

            if self._transaction.is_active:
                self._transaction.record_operation("create", entity.id, data)

            return self._dict_to_entity(data)

    def read(self, id: str) -> Optional[T]:
        """读取实体"""
        with self._lock:
            data = self._store.get(id)
            if data is None:
                return None
            return self._dict_to_entity(copy.deepcopy(data))

    def update(self, id: str, data: Dict[str, Any]) -> Optional[T]:
        """更新实体"""
        with self._lock:
            existing = self._store.get(id)
            if existing is None:
                return None

            # 不允许更新id和created_at
            update_data = {k: v for k, v in data.items() if k not in ("id", "created_at")}
            update_data["updated_at"] = time.time()

            # 深拷贝后更新
            updated = copy.deepcopy(existing)
            for key, value in update_data.items():
                if key == "metadata" and isinstance(value, dict) and isinstance(updated.get("metadata"), dict):
                    updated["metadata"].update(value)
                else:
                    updated[key] = value

            self._store[id] = updated

            if self._transaction.is_active:
                self._transaction.record_operation("update", id, copy.deepcopy(existing))

            return self._dict_to_entity(updated)

    def delete(self, id: str) -> bool:
        """删除实体"""
        with self._lock:
            if id not in self._store:
                return False

            removed = self._store.pop(id)

            if self._transaction.is_active:
                self._transaction.record_operation("delete", id, removed)

            return True

    def query(
        self,
        filters: Optional[List[QueryFilter]] = None,
        sort: Optional[List[SortOrder]] = None,
        pagination: Optional[Pagination] = None,
    ) -> QueryResult:
        """查询实体"""
        with self._lock:
            # 获取所有数据的副本
            all_data = list(self._store.values())

            # 应用过滤
            if filters:
                all_data = self._apply_filters(all_data, filters)

            # 计算总数（在排序和分页之前）
            total = len(all_data)

            # 应用排序
            if sort:
                all_data = self._apply_sort(all_data, sort)

            # 应用分页
            if pagination:
                offset = pagination.effective_offset
                limit = pagination.effective_limit
                page = pagination.page
                page_size = pagination.effective_limit
                paginated_data = all_data[offset:offset + limit]
            else:
                paginated_data = all_data
                page = 1
                page_size = len(all_data)

            # 转换为实体
            items = [self._dict_to_entity(d) for d in paginated_data]

            return QueryResult(
                items=items,
                total=total,
                page=page,
                page_size=page_size,
            )

    def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """计数"""
        with self._lock:
            if filters is None:
                return len(self._store)

            all_data = list(self._store.values())
            filtered = self._apply_filters(all_data, filters)
            return len(filtered)

    def exists(self, id: str) -> bool:
        """检查实体是否存在"""
        with self._lock:
            return id in self._store

    # ============================================================
    # 批量操作
    # ============================================================

    def bulk_create(self, entities: List[T]) -> List[T]:
        """
        批量创建实体

        Args:
            entities: 实体列表

        Returns:
            创建后的实体列表
        """
        with self._lock:
            results = []
            for entity in entities:
                if entity.id is None:
                    entity.id = self._generate_id()
                now = time.time()
                if entity.created_at is None:
                    entity.created_at = now
                if entity.updated_at is None:
                    entity.updated_at = now
                data = entity.to_dict()
                self._store[entity.id] = data
                results.append(self._dict_to_entity(copy.deepcopy(data)))
            return results

    def bulk_update(self, updates: List[Dict[str, Any]]) -> List[Optional[T]]:
        """
        批量更新实体

        Args:
            updates: 更新列表，每项包含id和要更新的字段

        Returns:
            更新后的实体列表（不存在的返回None）
        """
        with self._lock:
            results = []
            now = time.time()
            for update_data in updates:
                entity_id = update_data.pop("id", None)
                if entity_id is None:
                    results.append(None)
                    continue
                existing = self._store.get(entity_id)
                if existing is None:
                    results.append(None)
                    continue
                updated = copy.deepcopy(existing)
                for key, value in update_data.items():
                    if key not in ("id", "created_at"):
                        if key == "metadata" and isinstance(value, dict) and isinstance(updated.get("metadata"), dict):
                            updated["metadata"].update(value)
                        else:
                            updated[key] = value
                updated["updated_at"] = now
                self._store[entity_id] = updated
                results.append(self._dict_to_entity(updated))
            return results

    def bulk_delete(self, ids: List[str]) -> int:
        """
        批量删除实体

        Args:
            ids: 要删除的实体ID列表

        Returns:
            成功删除的数量
        """
        with self._lock:
            count = 0
            for entity_id in ids:
                if entity_id in self._store:
                    del self._store[entity_id]
                    count += 1
            return count

    # ============================================================
    # 事务支持
    # ============================================================

    def begin(self) -> Transaction:
        """开始事务"""
        self._transaction.begin()
        return self._transaction

    def commit(self) -> None:
        """提交当前事务"""
        self._transaction.commit()

    def rollback(self) -> None:
        """回滚当前事务"""
        self._transaction.rollback()

    @property
    def in_transaction(self) -> bool:
        """是否在事务中"""
        return self._transaction.is_active

    # ============================================================
    # 辅助方法
    # ============================================================

    def clear(self) -> None:
        """清空所有数据"""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """获取存储中的实体数量"""
        with self._lock:
            return len(self._store)

    def all_ids(self) -> List[str]:
        """获取所有实体ID"""
        with self._lock:
            return list(self._store.keys())

    def _generate_id(self) -> str:
        """生成唯一ID"""
        self._id_counter += 1
        return str(uuid.uuid4())

    def _dict_to_entity(self, data: Dict[str, Any]) -> T:
        """将字典转换为实体对象"""
        return self._entity_class.from_dict(data)

    def _apply_filters(
        self,
        data: List[Dict[str, Any]],
        filters: List[QueryFilter],
    ) -> List[Dict[str, Any]]:
        """应用过滤条件"""
        result = data
        for f in filters:
            result = [item for item in result if self._filter_matches(f, item)]
        return result

    def _filter_matches(self, f: QueryFilter, data: Dict[str, Any]) -> bool:
        """检查单条数据是否匹配过滤器"""
        field_value = self._get_nested_value(data, f.field)

        if f.operator == FilterOperator.EQ:
            return field_value == f.value
        elif f.operator == FilterOperator.NE:
            return field_value != f.value
        elif f.operator == FilterOperator.GT:
            if field_value is None:
                return False
            return field_value > f.value
        elif f.operator == FilterOperator.GTE:
            if field_value is None:
                return False
            return field_value >= f.value
        elif f.operator == FilterOperator.LT:
            if field_value is None:
                return False
            return field_value < f.value
        elif f.operator == FilterOperator.LTE:
            if field_value is None:
                return False
            return field_value <= f.value
        elif f.operator == FilterOperator.IN:
            if field_value is None:
                return False
            return field_value in f.value
        elif f.operator == FilterOperator.NOT_IN:
            if field_value is None:
                return True
            return field_value not in f.value
        elif f.operator == FilterOperator.CONTAINS:
            if field_value is None or f.value is None:
                return False
            return f.value in str(field_value)
        elif f.operator == FilterOperator.STARTSWITH:
            if field_value is None or f.value is None:
                return False
            return str(field_value).startswith(str(f.value))
        elif f.operator == FilterOperator.ENDSWITH:
            if field_value is None or f.value is None:
                return False
            return str(field_value).endswith(str(f.value))
        elif f.operator == FilterOperator.REGEX:
            if field_value is None:
                return False
            try:
                return bool(re.search(f.value, str(field_value)))
            except re.error:
                return False
        elif f.operator == FilterOperator.IS_NULL:
            return field_value is None
        elif f.operator == FilterOperator.IS_NOT_NULL:
            return field_value is not None
        elif f.operator == FilterOperator.BETWEEN:
            if field_value is None:
                return False
            low, high = f.value
            return low <= field_value <= high
        return False

    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """获取嵌套字段值"""
        parts = field.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current

    def _apply_sort(
        self,
        data: List[Dict[str, Any]],
        sort_orders: List[SortOrder],
    ) -> List[Dict[str, Any]]:
        """应用排序"""
        if not sort_orders:
            return data

        def sort_key(item: Dict[str, Any]) -> tuple:
            keys = []
            for order in sort_orders:
                value = self._get_nested_value(item, order.field)
                # 处理None值，确保None排在最后
                if value is None:
                    if order.direction == SortDirection.ASC:
                        keys.append((1, ""))
                    else:
                        keys.append((1, ""))
                elif isinstance(value, (int, float)):
                    if order.direction == SortDirection.DESC:
                        value = -value
                    keys.append((0, value))
                elif isinstance(value, str):
                    if order.direction == SortDirection.DESC:
                        # 反转字符串排序
                        keys.append((0, value))
                    else:
                        keys.append((0, value))
                elif isinstance(value, bool):
                    keys.append((0, int(value)))
                else:
                    keys.append((0, str(value)))
            return tuple(keys)

        # 对每个排序字段进行稳定排序（从最后一个字段开始，保证优先级）
        result = list(data)
        for order in reversed(sort_orders):
            reverse = order.direction == SortDirection.DESC
            result.sort(
                key=lambda item: self._sort_field_value(item, order.field),
                reverse=reverse,
            )
        return result

    def _sort_field_value(self, item: Dict[str, Any], field: str) -> Any:
        """获取排序字段的值，处理None和类型不一致的情况"""
        value = self._get_nested_value(item, field)
        if value is None:
            return (1, "")  # None排在最后
        if isinstance(value, (int, float)):
            return (0, value)
        if isinstance(value, str):
            return (0, value)
        if isinstance(value, bool):
            return (0, int(value))
        return (0, str(value))

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, id: str) -> bool:
        return self.exists(id)

    def __repr__(self) -> str:
        return f"<MemoryRepository entities={self.size()}>"
