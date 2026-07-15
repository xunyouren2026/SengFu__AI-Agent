"""
文件存储实现

基于JSON文件的持久化存储，支持索引加速查询、数据压缩、文件锁和数据备份恢复。
"""

import copy
import json
import os
import shutil
import struct
import threading
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar

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


class FileLock:
    """
    文件锁实现

    跨进程的文件锁，使用fcntl（Unix）或模拟实现。
    """

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._lock_file = filepath + ".lock"
        self._fd = None
        self._thread_lock = threading.Lock()
        self._use_fcntl = True
        try:
            import fcntl  # type: ignore
        except ImportError:
            self._use_fcntl = False

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        获取文件锁

        Args:
            timeout: 超时时间（秒）

        Returns:
            是否成功获取锁
        """
        with self._thread_lock:
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    self._fd = open(self._lock_file, "w")
                    if self._use_fcntl:
                        import fcntl  # type: ignore
                        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except (IOError, OSError):
                    if self._fd:
                        try:
                            self._fd.close()
                        except Exception:
                            pass
                        self._fd = None
                    time.sleep(0.05)
            return False

    def release(self) -> None:
        """释放文件锁"""
        with self._thread_lock:
            if self._fd:
                try:
                    if self._use_fcntl:
                        import fcntl  # type: ignore
                        fcntl.flock(self._fd, fcntl.LOCK_UN)
                    self._fd.close()
                except Exception:
                    pass
                finally:
                    self._fd = None
                try:
                    if os.path.exists(self._lock_file):
                        os.remove(self._lock_file)
                except Exception:
                    pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class Index:
    """
    内存索引

    加速特定字段的查询操作。
    """

    def __init__(self, field_name: str):
        self.field_name = field_name
        self._index: Dict[Any, Set[str]] = {}
        self._lock = threading.Lock()

    def add(self, entity_id: str, value: Any) -> None:
        """添加索引条目"""
        with self._lock:
            if value is not None:
                key = self._make_key(value)
                if key not in self._index:
                    self._index[key] = set()
                self._index[key].add(entity_id)

    def remove(self, entity_id: str, value: Any) -> None:
        """移除索引条目"""
        with self._lock:
            if value is not None:
                key = self._make_key(value)
                if key in self._index:
                    self._index[key].discard(entity_id)
                    if not self._index[key]:
                        del self._index[key]

    def lookup(self, value: Any) -> Set[str]:
        """查找匹配的实体ID"""
        with self._lock:
            key = self._make_key(value)
            return set(self._index.get(key, set()))

    def lookup_range(self, low: Any, high: Any) -> Set[str]:
        """范围查找"""
        with self._lock:
            result = set()
            low_key = self._make_key(low)
            high_key = self._make_key(high)
            for key, ids in self._index.items():
                if low_key <= key <= high_key:
                    result.update(ids)
            return result

    def clear(self) -> None:
        """清空索引"""
        with self._lock:
            self._index.clear()

    def _make_key(self, value: Any) -> Any:
        """创建索引键"""
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)


class FileRepository(Repository[T]):
    """
    文件存储实现

    基于JSON文件的持久化存储，支持：
    - 自动保存到磁盘
    - 内存索引加速查询
    - zlib数据压缩
    - 文件锁（跨进程安全）
    - 数据备份和恢复
    - 线程安全

    Args:
        filepath: 数据文件路径
        entity_class: 实体类
        compressed: 是否启用压缩
        auto_save: 是否自动保存
        index_fields: 需要建立索引的字段列表
    """

    def __init__(
        self,
        filepath: str,
        entity_class: Optional[Type[T]] = None,
        compressed: bool = False,
        auto_save: bool = True,
        index_fields: Optional[List[str]] = None,
    ):
        self._filepath = filepath
        self._entity_class = entity_class or Entity
        self._compressed = compressed
        self._auto_save = auto_save
        self._lock = threading.RLock()
        self._file_lock = FileLock(filepath)
        self._store: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._save_count = 0

        # 初始化索引
        self._indexes: Dict[str, Index] = {}
        if index_fields:
            for field_name in index_fields:
                self._indexes[field_name] = Index(field_name)

        # 确保目录存在
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # 加载已有数据
        self._load()

    # ============================================================
    # CRUD 操作
    # ============================================================

    def create(self, entity: T) -> T:
        """创建实体"""
        with self._lock:
            if entity.id is None:
                entity.id = str(uuid.uuid4())

            now = time.time()
            if entity.created_at is None:
                entity.created_at = now
            if entity.updated_at is None:
                entity.updated_at = now

            data = entity.to_dict()
            self._store[entity.id] = data
            self._update_indexes(entity.id, data, is_new=True)
            self._dirty = True

            if self._auto_save:
                self._save()

            return self._dict_to_entity(copy.deepcopy(data))

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

            # 移除旧索引
            self._remove_from_indexes(id, existing)

            update_data = {k: v for k, v in data.items() if k not in ("id", "created_at")}
            update_data["updated_at"] = time.time()

            updated = copy.deepcopy(existing)
            for key, value in update_data.items():
                if key == "metadata" and isinstance(value, dict) and isinstance(updated.get("metadata"), dict):
                    updated["metadata"].update(value)
                else:
                    updated[key] = value

            self._store[id] = updated

            # 添加新索引
            self._update_indexes(id, updated, is_new=False)

            self._dirty = True
            if self._auto_save:
                self._save()

            return self._dict_to_entity(updated)

    def delete(self, id: str) -> bool:
        """删除实体"""
        with self._lock:
            if id not in self._store:
                return False

            existing = self._store.pop(id)
            self._remove_from_indexes(id, existing)

            self._dirty = True
            if self._auto_save:
                self._save()

            return True

    def query(
        self,
        filters: Optional[List[QueryFilter]] = None,
        sort: Optional[List[SortOrder]] = None,
        pagination: Optional[Pagination] = None,
    ) -> QueryResult:
        """查询实体"""
        with self._lock:
            # 尝试使用索引优化查询
            candidate_ids = self._try_index_lookup(filters)

            if candidate_ids is not None:
                all_data = [
                    self._store[eid]
                    for eid in candidate_ids
                    if eid in self._store
                ]
            else:
                all_data = list(self._store.values())

            # 应用过滤
            if filters:
                all_data = self._apply_filters(all_data, filters)

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

            items = [self._dict_to_entity(d) for d in paginated_data]
            return QueryResult(items=items, total=total, page=page, page_size=page_size)

    def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """计数"""
        with self._lock:
            if filters is None:
                return len(self._store)

            candidate_ids = self._try_index_lookup(filters)
            if candidate_ids is not None:
                data = [self._store[eid] for eid in candidate_ids if eid in self._store]
            else:
                data = list(self._store.values())

            filtered = self._apply_filters(data, filters)
            return len(filtered)

    def exists(self, id: str) -> bool:
        """检查实体是否存在"""
        with self._lock:
            return id in self._store

    # ============================================================
    # 持久化
    # ============================================================

    def save(self) -> None:
        """手动保存到磁盘"""
        with self._lock:
            self._save()

    def _save(self) -> None:
        """内部保存实现"""
        with self._file_lock:
            json_str = json.dumps(self._store, ensure_ascii=False, default=str)
            if self._compressed:
                json_bytes = json_str.encode("utf-8")
                compressed = zlib.compress(json_bytes, level=6)
                with open(self._filepath, "wb") as f:
                    f.write(compressed)
            else:
                with open(self._filepath, "w", encoding="utf-8") as f:
                    f.write(json_str)
            self._dirty = False
            self._save_count += 1

    def _load(self) -> None:
        """从磁盘加载数据"""
        if not os.path.exists(self._filepath):
            self._store = {}
            return

        with self._file_lock:
            try:
                if self._compressed:
                    with open(self._filepath, "rb") as f:
                        compressed = f.read()
                    json_bytes = zlib.decompress(compressed)
                    json_str = json_bytes.decode("utf-8")
                else:
                    with open(self._filepath, "r", encoding="utf-8") as f:
                        json_str = f.read()

                self._store = json.loads(json_str)

                # 重建索引
                self._rebuild_indexes()
            except (json.JSONDecodeError, zlib.error, IOError):
                self._store = {}

    # ============================================================
    # 索引管理
    # ============================================================

    def add_index(self, field_name: str) -> None:
        """添加索引"""
        with self._lock:
            if field_name not in self._indexes:
                self._indexes[field_name] = Index(field_name)
                # 为已有数据建立索引
                for entity_id, data in self._store.items():
                    value = self._get_nested_value(data, field_name)
                    self._indexes[field_name].add(entity_id, value)

    def remove_index(self, field_name: str) -> None:
        """移除索引"""
        with self._lock:
            if field_name in self._indexes:
                self._indexes[field_name].clear()
                del self._indexes[field_name]

    def _update_indexes(self, entity_id: str, data: Dict[str, Any], is_new: bool) -> None:
        """更新索引"""
        for field_name, index in self._indexes.items():
            value = self._get_nested_value(data, field_name)
            index.add(entity_id, value)

    def _remove_from_indexes(self, entity_id: str, data: Dict[str, Any]) -> None:
        """从索引中移除"""
        for field_name, index in self._indexes.items():
            value = self._get_nested_value(data, field_name)
            index.remove(entity_id, value)

    def _rebuild_indexes(self) -> None:
        """重建所有索引"""
        for index in self._indexes.values():
            index.clear()
        for entity_id, data in self._store.items():
            self._update_indexes(entity_id, data, is_new=True)

    def _try_index_lookup(
        self, filters: Optional[List[QueryFilter]]
    ) -> Optional[Set[str]]:
        """尝试使用索引加速查询"""
        if not filters or not self._indexes:
            return None

        # 找到第一个可以使用索引的过滤器
        for f in filters:
            if f.field in self._indexes:
                index = self._indexes[f.field]
                if f.operator == FilterOperator.EQ:
                    return index.lookup(f.value)
                elif f.operator == FilterOperator.IN:
                    result = set()
                    for val in f.value:
                        result.update(index.lookup(val))
                    return result
                elif f.operator == FilterOperator.BETWEEN:
                    low, high = f.value
                    return index.lookup_range(low, high)

        return None

    # ============================================================
    # 备份与恢复
    # ============================================================

    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        创建数据备份

        Args:
            backup_path: 备份文件路径，为None时自动生成

        Returns:
            备份文件路径
        """
        with self._lock:
            if backup_path is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(self._filepath)
                backup_path = f"{base}_backup_{timestamp}{ext}"

            if os.path.exists(self._filepath):
                shutil.copy2(self._filepath, backup_path)
            else:
                # 空数据备份
                with open(backup_path, "w") as f:
                    json.dump({}, f)

            return backup_path

    def restore(self, backup_path: str) -> bool:
        """
        从备份恢复数据

        Args:
            backup_path: 备份文件路径

        Returns:
            是否恢复成功
        """
        with self._lock:
            if not os.path.exists(backup_path):
                return False

            try:
                if self._compressed:
                    with open(backup_path, "rb") as f:
                        compressed = f.read()
                    json_bytes = zlib.decompress(compressed)
                    json_str = json_bytes.decode("utf-8")
                else:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        json_str = f.read()

                data = json.loads(json_str)
                if not isinstance(data, dict):
                    return False

                self._store = data
                self._rebuild_indexes()
                self._save()
                return True
            except (json.JSONDecodeError, zlib.error, IOError):
                return False

    def list_backups(self) -> List[str]:
        """列出所有备份文件"""
        base, ext = os.path.splitext(self._filepath)
        dir_path = os.path.dirname(self._filepath) or "."
        prefix = os.path.basename(base) + "_backup_"
        backups = []
        if os.path.exists(dir_path):
            for fname in os.listdir(dir_path):
                if fname.startswith(prefix) and fname.endswith(ext):
                    backups.append(os.path.join(dir_path, fname))
        backups.sort(reverse=True)
        return backups

    # ============================================================
    # 辅助方法
    # ============================================================

    def _dict_to_entity(self, data: Dict[str, Any]) -> T:
        """字典转实体"""
        return self._entity_class.from_dict(data)

    def _apply_filters(
        self, data: List[Dict[str, Any]], filters: List[QueryFilter]
    ) -> List[Dict[str, Any]]:
        """应用过滤"""
        import re
        result = data
        for f in filters:
            filtered = []
            for item in result:
                field_value = self._get_nested_value(item, f.field)
                if self._matches_filter(field_value, f.operator, f.value):
                    filtered.append(item)
            result = filtered
        return result

    def _matches_filter(
        self, value: Any, operator: FilterOperator, target: Any
    ) -> bool:
        """检查值是否匹配过滤条件"""
        if operator == FilterOperator.EQ:
            return value == target
        elif operator == FilterOperator.NE:
            return value != target
        elif operator == FilterOperator.GT:
            return value is not None and value > target
        elif operator == FilterOperator.GTE:
            return value is not None and value >= target
        elif operator == FilterOperator.LT:
            return value is not None and value < target
        elif operator == FilterOperator.LTE:
            return value is not None and value <= target
        elif operator == FilterOperator.IN:
            return value is not None and value in target
        elif operator == FilterOperator.NOT_IN:
            return value is None or value not in target
        elif operator == FilterOperator.CONTAINS:
            return value is not None and target in str(value)
        elif operator == FilterOperator.STARTSWITH:
            return value is not None and str(value).startswith(str(target))
        elif operator == FilterOperator.ENDSWITH:
            return value is not None and str(value).endswith(str(target))
        elif operator == FilterOperator.REGEX:
            if value is None:
                return False
            try:
                return bool(re.search(target, str(value)))
            except re.error:
                return False
        elif operator == FilterOperator.IS_NULL:
            return value is None
        elif operator == FilterOperator.IS_NOT_NULL:
            return value is not None
        elif operator == FilterOperator.BETWEEN:
            if value is None:
                return False
            low, high = target
            return low <= value <= high
        return False

    def _apply_sort(
        self, data: List[Dict[str, Any]], sort_orders: List[SortOrder]
    ) -> List[Dict[str, Any]]:
        """应用排序"""
        result = list(data)
        for order in reversed(sort_orders):
            reverse = order.direction == SortDirection.DESC
            result.sort(
                key=lambda item, f=order.field: self._sort_key(item, f),
                reverse=reverse,
            )
        return result

    def _sort_key(self, item: Dict[str, Any], field: str) -> tuple:
        """排序键"""
        value = self._get_nested_value(item, field)
        if value is None:
            return (1, "")
        if isinstance(value, (int, float)):
            return (0, value)
        if isinstance(value, bool):
            return (0, int(value))
        return (0, str(value))

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

    def clear(self) -> None:
        """清空所有数据"""
        with self._lock:
            self._store.clear()
            for index in self._indexes.values():
                index.clear()
            self._dirty = True
            if self._auto_save:
                self._save()

    def size(self) -> int:
        """获取实体数量"""
        with self._lock:
            return len(self._store)

    def all_ids(self) -> List[str]:
        """获取所有实体ID"""
        with self._lock:
            return list(self._store.keys())

    @property
    def filepath(self) -> str:
        """数据文件路径"""
        return self._filepath

    @property
    def is_dirty(self) -> bool:
        """是否有未保存的更改"""
        return self._dirty

    @property
    def save_count(self) -> int:
        """保存次数"""
        return self._save_count

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return f"<FileRepository path={self._filepath!r} entities={self.size()}>"
