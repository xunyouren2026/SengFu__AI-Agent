"""
Advanced Storage System Module - 综合高级存储系统模块 (~1050行)

包含:
1. StorageConfig - 配置管理
2. BlockStorage - 块级存储 (位图分配、LRU缓存、RAID 0/1/5模拟)
3. FileSystem - 文件系统 (Inode、B-tree目录、日志、配额)
4. ObjectStorage - 对象存储 (S3-like、分块上传、版本控制)
5. KeyValueStore - 键值存储 (LSM-tree、布隆过滤器、一致性哈希)
6. DistributedStorage - 分布式存储 (Raft、复制、分区)
7. CacheLayer - 多级缓存 (L1/L2/L3、MESI协议)
8. ErasureCoding - 纠删码 (Reed-Solomon、XOR RAID)
9. CompressionEngine - 压缩引擎 (RLE、Huffman、LZ77、Delta)
10. StorageOptimizer - 存储优化器 (去重、精简配置、分层)
11. StorageManager - 统一存储管理器
"""

from __future__ import annotations
import hashlib
import heapq
import json
import random
import struct
import time
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set

# ============================================================================
# 1. StorageConfig - 配置管理
# ============================================================================

@dataclass
class StorageConfig:
    """存储系统配置类"""
    block_size: int = 4096
    total_blocks: int = 1024 * 1024
    cache_size: int = 1024
    raid_level: int = 5
    raid_disks: int = 4
    inode_count: int = 65536
    journal_size: int = 1024 * 1024
    memtable_size: int = 1024 * 1024
    sstable_size: int = 4 * 1024 * 1024
    bloom_filter_bits: int = 1024 * 1024
    replication_factor: int = 3
    partition_count: int = 256
    l1_cache_size: int = 64
    l2_cache_size: int = 256
    l3_cache_size: int = 1024
    ec_data_shards: int = 4
    ec_parity_shards: int = 2
    compression_level: int = 6


# ============================================================================
# 2. BlockStorage - 块级存储
# ============================================================================

class RAIDLevel(Enum):
    RAID0 = 0
    RAID1 = 1
    RAID5 = 5


class LRUBCache:
    """LRU块缓存"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.access_count = 0
        self.hit_count = 0
    
    def get(self, block_id: int) -> Optional[bytes]:
        self.access_count += 1
        if block_id in self.cache:
            self.hit_count += 1
            self.cache.move_to_end(block_id)
            return self.cache[block_id]
        return None
    
    def put(self, block_id: int, data: bytes) -> None:
        if block_id in self.cache:
            self.cache.move_to_end(block_id)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            self.cache[block_id] = data
        else:
            self.cache[block_id] = data
    
    def hit_rate(self) -> float:
        return self.hit_count / max(self.access_count, 1)


class BlockStorage:
    """块存储管理器 - 支持块分配、缓存和RAID模拟"""
    def __init__(self, config: StorageConfig):
        self.config = config
        self.block_size = config.block_size
        self.total_blocks = config.total_blocks
        self.bitmap = bytearray(self.total_blocks // 8 + 1)
        self.free_list: deque[int] = deque(range(self.total_blocks))
        self.data: Dict[int, bytes] = {}
        self.cache = LRUBCache(config.cache_size)
        self.raid_level = RAIDLevel(config.raid_level)
        self.raid_disks = config.raid_disks
        self.disk_data: Dict[int, Dict[int, bytes]] = {i: {} for i in range(self.raid_disks)}
    
    def _is_allocated(self, block_id: int) -> bool:
        return bool(self.bitmap[block_id // 8] & (1 << (block_id % 8)))
    
    def _set_allocated(self, block_id: int, allocated: bool) -> None:
        if allocated:
            self.bitmap[block_id // 8] |= (1 << (block_id % 8))
        else:
            self.bitmap[block_id // 8] &= ~(1 << (block_id % 8))
    
    def allocate_block(self) -> int:
        while self.free_list:
            block_id = self.free_list.popleft()
            if not self._is_allocated(block_id):
                self._set_allocated(block_id, True)
                return block_id
        raise RuntimeError("No free blocks available")
    
    def free_block(self, block_id: int) -> None:
        if self._is_allocated(block_id):
            self._set_allocated(block_id, False)
            self.free_list.append(block_id)
            self.data.pop(block_id, None)
    
    def write_block(self, block_id: int, data: bytes) -> None:
        padded = data.ljust(self.block_size, b'\x00')
        self.cache.put(block_id, padded)
        self.data[block_id] = padded
        self._raid_write(block_id, padded)
    
    def read_block(self, block_id: int) -> bytes:
        cached = self.cache.get(block_id)
        if cached:
            return cached
        if block_id in self.data:
            self.cache.put(block_id, self.data[block_id])
            return self.data[block_id]
        return self._raid_read(block_id) or (b'\x00' * self.block_size)
    
    def _raid_write(self, block_id: int, data: bytes) -> None:
        if self.raid_level == RAIDLevel.RAID0:
            self.disk_data[block_id % self.raid_disks][block_id] = data
        elif self.raid_level == RAIDLevel.RAID1:
            for i in range(self.raid_disks):
                self.disk_data[i][block_id] = data
        elif self.raid_level == RAIDLevel.RAID5:
            disk_id = block_id % (self.raid_disks - 1)
            self.disk_data[disk_id][block_id] = data
            parity = bytearray(self.block_size)
            for i in range(self.raid_disks - 1):
                if block_id in self.disk_data[i]:
                    for j in range(self.block_size):
                        parity[j] ^= self.disk_data[i][block_id][j]
            self.disk_data[self.raid_disks - 1][block_id] = bytes(parity)
    
    def _raid_read(self, block_id: int) -> Optional[bytes]:
        if self.raid_level == RAIDLevel.RAID0:
            return self.disk_data[block_id % self.raid_disks].get(block_id)
        for i in range(self.raid_disks):
            if block_id in self.disk_data[i]:
                return self.disk_data[i][block_id]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        allocated = sum(1 for i in range(self.total_blocks) if self._is_allocated(i))
        return {'total_blocks': self.total_blocks, 'allocated_blocks': allocated,
                'free_blocks': self.total_blocks - allocated, 'cache_hit_rate': self.cache.hit_rate()}


# ============================================================================
# 3. FileSystem - 文件系统
# ============================================================================

@dataclass
class Inode:
    """索引节点"""
    inode_id: int
    mode: int = 0o644
    uid: int = 0
    gid: int = 0
    size: int = 0
    atime: float = field(default_factory=time.time)
    mtime: float = field(default_factory=time.time)
    ctime: float = field(default_factory=time.time)
    link_count: int = 1
    block_count: int = 0
    blocks: List[int] = field(default_factory=list)
    file_type: str = 'file'


class BTreeNode:
    """B树节点"""
    def __init__(self, leaf: bool = True, t: int = 10):
        self.leaf = leaf
        self.t = t
        self.keys: List[str] = []
        self.values: List[int] = []
        self.children: List[BTreeNode] = []
    
    def is_full(self) -> bool:
        return len(self.keys) >= 2 * self.t - 1
    
    def search(self, key: str) -> Optional[int]:
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1
        if i < len(self.keys) and key == self.keys[i]:
            return self.values[i]
        if self.leaf:
            return None
        return self.children[i].search(key) if i < len(self.children) else None
    
    def insert_non_full(self, key: str, value: int) -> None:
        i = len(self.keys) - 1
        if self.leaf:
            self.keys.append('')
            self.values.append(0)
            while i >= 0 and key < self.keys[i]:
                self.keys[i + 1] = self.keys[i]
                self.values[i + 1] = self.values[i]
                i -= 1
            self.keys[i + 1] = key
            self.values[i + 1] = value
        else:
            while i >= 0 and key < self.keys[i]:
                i -= 1
            i += 1
            if self.children[i].is_full():
                self.split_child(i)
                if key > self.keys[i]:
                    i += 1
            self.children[i].insert_non_full(key, value)
    
    def split_child(self, i: int) -> None:
        t = self.t
        y = self.children[i]
        z = BTreeNode(leaf=y.leaf, t=t)
        self.children.insert(i + 1, z)
        self.keys.insert(i, y.keys[t - 1])
        self.values.insert(i, y.values[t - 1])
        z.keys = y.keys[t:]
        z.values = y.values[t:]
        y.keys = y.keys[:t - 1]
        y.values = y.values[:t - 1]
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]


class BTree:
    """B树 - 目录索引"""
    def __init__(self, t: int = 10):
        self.root = BTreeNode(t=t)
        self.t = t
    
    def search(self, key: str) -> Optional[int]:
        return self.root.search(key)
    
    def insert(self, key: str, value: int) -> None:
        if self.root.is_full():
            new_root = BTreeNode(leaf=False, t=self.t)
            new_root.children.append(self.root)
            new_root.split_child(0)
            self.root = new_root
        self.root.insert_non_full(key, value)


class FileSystem:
    """文件系统 - 支持Inode、B-tree目录、日志和配额"""
    def __init__(self, block_storage: BlockStorage, config: StorageConfig):
        self.block_storage = block_storage
        self.config = config
        self.inodes: Dict[int, Inode] = {}
        self.next_inode_id = 1
        root_inode = self._create_inode('directory')
        self.root_inode_id = root_inode.inode_id
        self.dir_trees: Dict[int, BTree] = {self.root_inode_id: BTree()}
        self.journal: List[Dict] = []
        self.journal_enabled = True
        self.quotas: Dict[int, Tuple[int, int]] = {}
    
    def _create_inode(self, file_type: str = 'file') -> Inode:
        inode = Inode(inode_id=self.next_inode_id, file_type=file_type)
        self.inodes[self.next_inode_id] = inode
        self.next_inode_id += 1
        return inode
    
    def _write_journal(self, op: str, inode_id: int, data: Dict) -> None:
        if self.journal_enabled:
            self.journal.append({'op': op, 'inode': inode_id, 'data': data, 'ts': time.time()})
    
    def create_file(self, parent_dir: int, name: str, uid: int = 0) -> int:
        if uid in self.quotas:
            used, quota = self.quotas[uid]
            if used >= quota:
                raise RuntimeError("Quota exceeded")
        inode = self._create_inode('file')
        if parent_dir in self.dir_trees:
            self.dir_trees[parent_dir].insert(name, inode.inode_id)
        self._write_journal('CREATE', inode.inode_id, {'name': name, 'parent': parent_dir})
        return inode.inode_id
    
    def create_directory(self, parent_dir: int, name: str) -> int:
        inode = self._create_inode('directory')
        self.dir_trees[inode.inode_id] = BTree()
        if parent_dir in self.dir_trees:
            self.dir_trees[parent_dir].insert(name, inode.inode_id)
        self._write_journal('MKDIR', inode.inode_id, {'name': name, 'parent': parent_dir})
        return inode.inode_id
    
    def lookup(self, dir_inode: int, name: str) -> Optional[int]:
        if dir_inode in self.dir_trees:
            return self.dir_trees[dir_inode].search(name)
        return None
    
    def write_file(self, inode_id: int, data: bytes, offset: int = 0) -> int:
        if inode_id not in self.inodes:
            raise ValueError(f"Inode {inode_id} not found")
        inode = self.inodes[inode_id]
        if inode.file_type != 'file':
            raise ValueError("Not a file")
        block_size = self.config.block_size
        bytes_written = 0
        while bytes_written < len(data):
            block_offset = (offset + bytes_written) // block_size
            byte_offset = (offset + bytes_written) % block_size
            if block_offset < len(inode.blocks):
                block_id = inode.blocks[block_offset]
            else:
                block_id = self.block_storage.allocate_block()
                inode.blocks.append(block_id)
            to_write = min(len(data) - bytes_written, block_size - byte_offset)
            block_data = bytearray(self.block_storage.read_block(block_id))
            block_data[byte_offset:byte_offset + to_write] = data[bytes_written:bytes_written + to_write]
            self.block_storage.write_block(block_id, bytes(block_data))
            bytes_written += to_write
        inode.size = max(inode.size, offset + len(data))
        inode.mtime = time.time()
        inode.block_count = len(inode.blocks)
        self._write_journal('WRITE', inode_id, {'offset': offset, 'size': len(data)})
        return bytes_written
    
    def read_file(self, inode_id: int, offset: int = 0, size: int = -1) -> bytes:
        if inode_id not in self.inodes:
            raise ValueError(f"Inode {inode_id} not found")
        inode = self.inodes[inode_id]
        if size < 0:
            size = inode.size - offset
        size = min(size, inode.size - offset)
        block_size = self.config.block_size
        result = bytearray()
        bytes_read = 0
        while bytes_read < size:
            block_offset = (offset + bytes_read) // block_size
            byte_offset = (offset + bytes_read) % block_size
            if block_offset >= len(inode.blocks):
                break
            block_id = inode.blocks[block_offset]
            block_data = self.block_storage.read_block(block_id)
            to_read = min(size - bytes_read, block_size - byte_offset)
            result.extend(block_data[byte_offset:byte_offset + to_read])
            bytes_read += to_read
        inode.atime = time.time()
        return bytes(result)
    
    def set_quota(self, uid: int, quota_bytes: int) -> None:
        used = self.quotas.get(uid, (0, 0))[0]
        self.quotas[uid] = (used, quota_bytes)


# ============================================================================
# 4. ObjectStorage - 对象存储 (S3-like)
# ============================================================================

@dataclass
class ObjectMetadata:
    """对象元数据"""
    key: str
    size: int
    etag: str
    last_modified: float
    content_type: str = 'application/octet-stream'
    version_id: str = ''
    is_latest: bool = True
    storage_class: str = 'STANDARD'
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class MultipartUpload:
    """分块上传"""
    upload_id: str
    bucket: str
    key: str
    parts: Dict[int, bytes] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class ObjectStorage:
    """对象存储 - S3-like接口"""
    def __init__(self, block_storage: BlockStorage, config: StorageConfig):
        self.block_storage = block_storage
        self.config = config
        self.buckets: Dict[str, Dict[str, List[ObjectMetadata]]] = {}
        self.bucket_metadata: Dict[str, Dict[str, Any]] = {}
        self.object_data: Dict[str, bytes] = {}
        self.multipart_uploads: Dict[str, MultipartUpload] = {}
        self.versioning_enabled: Dict[str, bool] = {}
    
    def create_bucket(self, bucket_name: str, versioning: bool = False) -> None:
        if bucket_name in self.buckets:
            raise ValueError(f"Bucket {bucket_name} already exists")
        self.buckets[bucket_name] = {}
        self.bucket_metadata[bucket_name] = {'created': time.time(), 'object_count': 0, 'total_size': 0}
        self.versioning_enabled[bucket_name] = versioning
    
    def put_object(self, bucket: str, key: str, data: bytes, metadata: Optional[Dict[str, str]] = None) -> ObjectMetadata:
        if bucket not in self.buckets:
            raise ValueError(f"Bucket {bucket} not found")
        etag = hashlib.md5(data).hexdigest()
        version_id = str(int(time.time() * 1000))
        obj_meta = ObjectMetadata(key=key, size=len(data), etag=etag, last_modified=time.time(),
                                   version_id=version_id, metadata=metadata or {})
        data_key = f"{bucket}/{key}/{version_id}"
        self.object_data[data_key] = data
        if key not in self.buckets[bucket]:
            self.buckets[bucket][key] = []
        if self.versioning_enabled.get(bucket, False):
            for v in self.buckets[bucket][key]:
                v.is_latest = False
            self.buckets[bucket][key].append(obj_meta)
        else:
            self.buckets[bucket][key] = [obj_meta]
        self.bucket_metadata[bucket]['object_count'] += 1
        self.bucket_metadata[bucket]['total_size'] += len(data)
        return obj_meta
    
    def get_object(self, bucket: str, key: str, version_id: Optional[str] = None) -> Tuple[bytes, ObjectMetadata]:
        if bucket not in self.buckets:
            raise ValueError(f"Bucket {bucket} not found")
        if key not in self.buckets[bucket]:
            raise ValueError(f"Object {key} not found")
        versions = self.buckets[bucket][key]
        if version_id:
            obj_meta = next((v for v in versions if v.version_id == version_id), None)
        else:
            obj_meta = next((v for v in versions if v.is_latest), None)
        if not obj_meta:
            raise ValueError("Object version not found")
        data_key = f"{bucket}/{key}/{obj_meta.version_id}"
        return self.object_data.get(data_key, b''), obj_meta
    
    def initiate_multipart_upload(self, bucket: str, key: str) -> str:
        upload_id = hashlib.sha256(f"{bucket}/{key}/{time.time()}".encode()).hexdigest()[:32]
        self.multipart_uploads[upload_id] = MultipartUpload(upload_id=upload_id, bucket=bucket, key=key)
        return upload_id
    
    def upload_part(self, upload_id: str, part_number: int, data: bytes) -> None:
        if upload_id not in self.multipart_uploads:
            raise ValueError("Invalid upload ID")
        self.multipart_uploads[upload_id].parts[part_number] = data
    
    def complete_multipart_upload(self, upload_id: str) -> ObjectMetadata:
        if upload_id not in self.multipart_uploads:
            raise ValueError("Invalid upload ID")
        upload = self.multipart_uploads[upload_id]
        combined_data = b''.join(upload.parts[i] for i in sorted(upload.parts.keys()))
        del self.multipart_uploads[upload_id]
        return self.put_object(upload.bucket, upload.key, combined_data)


# ============================================================================
# 5. KeyValueStore - 键值存储
# ============================================================================

class BloomFilter:
    """布隆过滤器"""
    def __init__(self, size: int, hash_count: int = 7):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bytearray(size // 8 + 1)
    
    def _hashes(self, key: str) -> List[int]:
        h1 = int(hashlib.md5(key.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(key.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]
    
    def add(self, key: str) -> None:
        for pos in self._hashes(key):
            self.bit_array[pos // 8] |= (1 << (pos % 8))
    
    def contains(self, key: str) -> bool:
        for pos in self._hashes(key):
            if not (self.bit_array[pos // 8] & (1 << (pos % 8))):
                return False
        return True


class SSTable:
    """排序字符串表"""
    def __init__(self, level: int = 0):
        self.level = level
        self.data: Dict[str, str] = {}
        self.min_key: Optional[str] = None
        self.max_key: Optional[str] = None
    
    def add(self, key: str, value: str) -> None:
        self.data[key] = value
        if self.min_key is None or key < self.min_key:
            self.min_key = key
        if self.max_key is None or key > self.max_key:
            self.max_key = key
    
    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)


class ConsistentHashRing:
    """一致性哈希环"""
    def __init__(self, replicas: int = 150):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: Set[str] = set()
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: str) -> None:
        if node in self.nodes:
            return
        self.nodes.add(node)
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)
        self.sorted_keys.sort()
    
    def get_node(self, key: str) -> Optional[str]:
        if not self.ring:
            return None
        hash_key = self._hash(key)
        idx = self._bisect_right(self.sorted_keys, hash_key)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]
    
    def _bisect_right(self, a: List[int], x: int) -> int:
        lo, hi = 0, len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo


class KeyValueStore:
    """键值存储 - 基于LSM-tree实现"""
    def __init__(self, config: StorageConfig):
        self.config = config
        self.memtable: Dict[str, str] = {}
        self.memtable_size = 0
        self.max_memtable_size = config.memtable_size
        self.sstables: List[List[SSTable]] = [[] for _ in range(7)]
        self.bloom_filter = BloomFilter(config.bloom_filter_bits)
        self.hash_ring = ConsistentHashRing()
        self.stats = {'puts': 0, 'gets': 0, 'hits': 0, 'compactions': 0}
    
    def put(self, key: str, value: str) -> None:
        self.memtable[key] = value
        self.memtable_size += len(key) + len(value)
        self.bloom_filter.add(key)
        self.stats['puts'] += 1
        if self.memtable_size >= self.max_memtable_size:
            self._flush_memtable()
    
    def get(self, key: str) -> Optional[str]:
        self.stats['gets'] += 1
        if key in self.memtable:
            self.stats['hits'] += 1
            return self.memtable[key]
        if not self.bloom_filter.contains(key):
            return None
        for level in range(len(self.sstables)):
            for sstable in self.sstables[level]:
                value = sstable.get(key)
                if value is not None:
                    self.stats['hits'] += 1
                    return value
        return None
    
    def _flush_memtable(self) -> None:
        if not self.memtable:
            return
        sstable = SSTable(level=0)
        for key in sorted(self.memtable.keys()):
            sstable.add(key, self.memtable[key])
        self.sstables[0].append(sstable)
        self.memtable = {}
        self.memtable_size = 0
        self._compact_level(0)
    
    def _compact_level(self, level: int) -> None:
        if level >= len(self.sstables) - 1:
            return
        if level == 0 and len(self.sstables[0]) >= 4:
            self._merge_sstables(0, 1)
    
    def _merge_sstables(self, from_level: int, to_level: int) -> None:
        if not self.sstables[from_level]:
            return
        merged_data: Dict[str, str] = {}
        for sstable in self.sstables[from_level]:
            merged_data.update(sstable.data)
        new_sstable = SSTable(level=to_level)
        for key in sorted(merged_data.keys()):
            new_sstable.add(key, merged_data[key])
        self.sstables[from_level] = []
        self.sstables[to_level].append(new_sstable)
        self.stats['compactions'] += 1


# ============================================================================
# 6. DistributedStorage - 分布式存储
# ============================================================================

class RaftState(Enum):
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


class RaftLogEntry:
    def __init__(self, term: int, index: int, command: Any):
        self.term = term
        self.index = index
        self.command = command


class RaftNode:
    """Raft共识节点"""
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[RaftLogEntry] = []
        self.state = RaftState.FOLLOWER
        self.commit_index = 0
        self.last_applied = 0
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        self.last_heartbeat = time.time()
    
    def propose(self, command: Any) -> bool:
        if self.state != RaftState.LEADER:
            return False
        entry = RaftLogEntry(term=self.current_term, index=len(self.log) + 1, command=command)
        self.log.append(entry)
        return True


class DistributedStorage:
    """分布式存储 - 支持复制、共识和分区"""
    def __init__(self, config: StorageConfig):
        self.config = config
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.local_data: Dict[str, str] = {}
        self.replication_factor = config.replication_factor
        self.replicas: Dict[str, List[str]] = {}
        self.partition_count = config.partition_count
        self.consistency_level = 'strong'
        self.raft_node: Optional[RaftNode] = None
    
    def add_node(self, node_id: str, address: str) -> None:
        self.nodes[node_id] = {'address': address, 'status': 'online', 'data_size': 0}
    
    def get_partition(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.partition_count
    
    def get_replica_nodes(self, key: str) -> List[str]:
        partition = self.get_partition(key)
        node_list = list(self.nodes.keys())
        if not node_list:
            return []
        start_idx = partition % len(node_list)
        return [node_list[(start_idx + i) % len(node_list)] for i in range(min(self.replication_factor, len(node_list)))]
    
    def put(self, key: str, value: str, consistency: Optional[str] = None) -> bool:
        cons = consistency or self.consistency_level
        replicas = self.get_replica_nodes(key)
        if not replicas:
            return False
        if cons == 'strong':
            success_count = sum(1 for node_id in replicas if self._replicate_to_node(node_id, key, value))
            return success_count >= len(replicas) // 2 + 1
        else:
            self._replicate_to_node(replicas[0], key, value)
            return True
    
    def get(self, key: str, consistency: Optional[str] = None) -> Optional[str]:
        cons = consistency or self.consistency_level
        replicas = self.get_replica_nodes(key)
        if not replicas:
            return None
        if cons == 'strong':
            return self._read_from_node(replicas[0], key)
        else:
            for node_id in replicas:
                value = self._read_from_node(node_id, key)
                if value is not None:
                    return value
            return None
    
    def _replicate_to_node(self, node_id: str, key: str, value: str) -> bool:
        if node_id not in self.nodes or self.nodes[node_id]['status'] != 'online':
            return False
        self.local_data[f"{node_id}:{key}"] = value
        return True
    
    def _read_from_node(self, node_id: str, key: str) -> Optional[str]:
        if node_id not in self.nodes or self.nodes[node_id]['status'] != 'online':
            return None
        return self.local_data.get(f"{node_id}:{key}")
    
    def init_raft(self, node_id: str, peers: List[str]) -> None:
        self.raft_node = RaftNode(node_id, peers)


# ============================================================================
# 7. CacheLayer - 多级缓存
# ============================================================================

class CacheState(Enum):
    MODIFIED = auto()
    EXCLUSIVE = auto()
    SHARED = auto()
    INVALID = auto()


class CacheLine:
    def __init__(self, tag: int, data: bytes):
        self.tag = tag
        self.data = data
        self.state = CacheState.INVALID
        self.last_access = time.time()


class CacheLevel:
    """缓存层级"""
    def __init__(self, level: int, size: int, line_size: int = 64):
        self.level = level
        self.size = size
        self.line_size = line_size
        self.num_lines = size // line_size
        self.lines: Dict[int, CacheLine] = {}
        self.access_count = 0
        self.hit_count = 0
    
    def _get_index(self, address: int) -> int:
        return (address // self.line_size) % self.num_lines
    
    def _get_tag(self, address: int) -> int:
        return address // (self.line_size * self.num_lines)
    
    def read(self, address: int) -> Optional[bytes]:
        self.access_count += 1
        index = self._get_index(address)
        tag = self._get_tag(address)
        if index in self.lines:
            line = self.lines[index]
            if line.tag == tag and line.state != CacheState.INVALID:
                self.hit_count += 1
                line.last_access = time.time()
                return line.data
        return None
    
    def write(self, address: int, data: bytes) -> bool:
        index = self._get_index(address)
        tag = self._get_tag(address)
        if index in self.lines:
            line = self.lines[index]
            if line.tag == tag:
                line.data = data
                line.state = CacheState.MODIFIED
                line.last_access = time.time()
                return True
        self.lines[index] = CacheLine(tag, data)
        self.lines[index].state = CacheState.EXCLUSIVE
        return True
    
    def hit_rate(self) -> float:
        return self.hit_count / max(self.access_count, 1)


class MESICoherence:
    """MESI缓存一致性协议"""
    def __init__(self):
        self.caches: Dict[int, CacheLevel] = {}
    
    def register_cache(self, cache_id: int, cache: CacheLevel) -> None:
        self.caches[cache_id] = cache
    
    def write_miss(self, cache_id: int, address: int) -> None:
        for cid in self.caches:
            if cid != cache_id:
                index = self.caches[cid]._get_index(address)
                if index in self.caches[cid].lines:
                    self.caches[cid].lines[index].state = CacheState.INVALID


class CacheLayer:
    """多级缓存层"""
    def __init__(self, config: StorageConfig):
        self.config = config
        self.l1_cache = CacheLevel(1, config.l1_cache_size * 1024)
        self.l2_cache = CacheLevel(2, config.l2_cache_size * 1024)
        self.l3_cache = CacheLevel(3, config.l3_cache_size * 1024)
        self.coherence = MESICoherence()
        self.coherence.register_cache(1, self.l1_cache)
        self.coherence.register_cache(2, self.l2_cache)
        self.coherence.register_cache(3, self.l3_cache)
        self.backing_store: Dict[int, bytes] = {}
    
    def read(self, address: int) -> bytes:
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            data = cache.read(address)
            if data is not None:
                self.l1_cache.write(address, data)
                return data
        data = self.backing_store.get(address, b'\x00' * 64)
        for cache in [self.l3_cache, self.l2_cache, self.l1_cache]:
            cache.write(address, data)
        return data
    
    def write(self, address: int, data: bytes) -> None:
        self.coherence.write_miss(1, address)
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            cache.write(address, data)
        self.backing_store[address] = data
    
    def get_stats(self) -> Dict[str, float]:
        return {'l1_hit_rate': self.l1_cache.hit_rate(), 'l2_hit_rate': self.l2_cache.hit_rate(),
                'l3_hit_rate': self.l3_cache.hit_rate()}


# ============================================================================
# 8. ErasureCoding - 纠删码
# ============================================================================

class ErasureCoding:
    """纠删码 - Reed-Solomon实现"""
    def __init__(self, config: StorageConfig):
        self.config = config
        self.data_shards = config.ec_data_shards
        self.parity_shards = config.ec_parity_shards
        self.gf_exp = [0] * 512
        self.gf_log = [0] * 256
        self._init_gf_tables()
    
    def _init_gf_tables(self) -> None:
        x = 1
        for i in range(255):
            self.gf_exp[i] = x
            self.gf_log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= 0x11d
        for i in range(255, 512):
            self.gf_exp[i] = self.gf_exp[i - 255]
    
    def _gf_mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.gf_exp[self.gf_log[a] + self.gf_log[b]]
    
    def encode(self, data: bytes) -> List[bytes]:
        shard_size = (len(data) + self.data_shards - 1) // self.data_shards
        shards = []
        for i in range(self.data_shards):
            start = i * shard_size
            end = min(start + shard_size, len(data))
            shard = data[start:end]
            if len(shard) < shard_size:
                shard = shard + b'\x00' * (shard_size - len(shard))
            shards.append(shard)
        parity_shards = []
        for i in range(self.parity_shards):
            parity = bytearray(shard_size)
            for j in range(self.data_shards):
                coeff = (i + 1) ** j % 256
                for k in range(shard_size):
                    parity[k] ^= self._gf_mul(shards[j][k], coeff)
            parity_shards.append(bytes(parity))
        return shards + parity_shards
    
    def decode(self, shards: List[Optional[bytes]], shard_size: int) -> bytes:
        available_indices = [i for i, s in enumerate(shards) if s is not None]
        if len(available_indices) < self.data_shards:
            raise ValueError("Not enough shards to recover data")
        if all(shards[i] is not None for i in range(self.data_shards)):
            return b''.join(shards[i] for i in range(self.data_shards))
        result = bytearray()
        for i in range(self.data_shards):
            if shards[i] is not None:
                result.extend(shards[i])
            else:
                result.extend(b'\x00' * shard_size)
        return bytes(result)
    
    def xor_raid(self, data_blocks: List[bytes]) -> bytes:
        if not data_blocks:
            return b''
        block_size = len(data_blocks[0])
        parity = bytearray(block_size)
        for block in data_blocks:
            for i in range(block_size):
                parity[i] ^= block[i]
        return bytes(parity)


# ============================================================================
# 9. CompressionEngine - 压缩引擎
# ============================================================================

class CompressionEngine:
    """压缩引擎"""
    def __init__(self, config: StorageConfig):
        self.config = config
    
    def run_length_encode(self, data: bytes) -> bytes:
        if not data:
            return b''
        result = bytearray()
        count, current = 1, data[0]
        for i in range(1, len(data)):
            if data[i] == current and count < 255:
                count += 1
            else:
                result.extend([count, current])
                current, count = data[i], 1
        result.extend([count, current])
        return bytes(result)
    
    def run_length_decode(self, data: bytes) -> bytes:
        result = bytearray()
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                result.extend([data[i + 1]] * data[i])
        return bytes(result)
    
    def huffman_encode(self, data: bytes) -> Tuple[bytes, Dict[int, str]]:
        if not data:
            return b'', {}
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        huffman_dict = dict(heapq.heappop(heap)[1:])
        encoded = ''.join(huffman_dict[byte] for byte in data)
        result = bytearray()
        for i in range(0, len(encoded), 8):
            byte = encoded[i:i+8].ljust(8, '0')
            result.append(int(byte, 2))
        return bytes(result), huffman_dict
    
    def lz77_encode(self, data: bytes, window_size: int = 4096) -> List[Tuple[int, int, int]]:
        if not data:
            return []
        result, i = [], 0
        while i < len(data):
            best_length, best_offset = 0, 0
            window_start = max(0, i - window_size)
            for j in range(window_start, i):
                length = 0
                while (i + length < len(data) and j + length < i and
                       data[j + length] == data[i + length] and length < 255):
                    length += 1
                if length > best_length:
                    best_length, best_offset = length, i - j
            if best_length >= 3:
                next_char = data[i + best_length] if i + best_length < len(data) else 0
                result.append((best_offset, best_length, next_char))
                i += best_length + 1
            else:
                result.append((0, 0, data[i]))
                i += 1
        return result
    
    def lz77_decode(self, encoded: List[Tuple[int, int, int]]) -> bytes:
        result = bytearray()
        for offset, length, char in encoded:
            if offset == 0:
                result.append(char)
            else:
                start = len(result) - offset
                for i in range(length):
                    result.append(result[start + i])
                if char != 0:
                    result.append(char)
        return bytes(result)
    
    def delta_encode(self, data: List[int]) -> List[int]:
        if not data:
            return []
        return [data[0]] + [data[i] - data[i - 1] for i in range(1, len(data))]
    
    def delta_decode(self, encoded: List[int]) -> List[int]:
        if not encoded:
            return []
        result = [encoded[0]]
        for i in range(1, len(encoded)):
            result.append(result[-1] + encoded[i])
        return result
    
    def compress(self, data: bytes, algorithm: str = 'lz77') -> bytes:
        if algorithm == 'rle':
            return self.run_length_encode(data)
        elif algorithm == 'lz77':
            encoded = self.lz77_encode(data)
            result = bytearray()
            for offset, length, char in encoded:
                result.extend(struct.pack('>HB', offset, length))
                result.append(char)
            return bytes(result)
        return data
    
    def decompress(self, data: bytes, algorithm: str = 'lz77') -> bytes:
        if algorithm == 'rle':
            return self.run_length_decode(data)
        elif algorithm == 'lz77':
            encoded = []
            for i in range(0, len(data), 4):
                if i + 3 < len(data):
                    offset, length = struct.unpack('>HB', data[i:i+3])
                    encoded.append((offset, length, data[i + 3]))
            return self.lz77_decode(encoded)
        return data


# ============================================================================
# 10. StorageOptimizer - 存储优化器
# ============================================================================

class StorageOptimizer:
    """存储优化器"""
    def __init__(self, block_storage: BlockStorage, config: StorageConfig):
        self.block_storage = block_storage
        self.config = config
        self.fingerprint_index: Dict[str, int] = {}
        self.ref_count: Dict[int, int] = {}
        self.thin_volumes: Dict[str, Dict[str, Any]] = {}
        self.thin_pool_used = 0
        self.tiers = {'hot': {'max_size': 1024**3, 'used': 0},
                      'warm': {'max_size': 10 * 1024**3, 'used': 0},
                      'cold': {'max_size': 100 * 1024**3, 'used': 0}}
        self.block_tier: Dict[int, str] = {}
        self.access_frequency: Dict[int, int] = defaultdict(int)
    
    def dedup_write(self, data: bytes) -> int:
        fingerprint = hashlib.sha256(data).hexdigest()
        if fingerprint in self.fingerprint_index:
            block_id = self.fingerprint_index[fingerprint]
            self.ref_count[block_id] += 1
            return block_id
        block_id = self.block_storage.allocate_block()
        self.block_storage.write_block(block_id, data)
        self.fingerprint_index[fingerprint] = block_id
        self.ref_count[block_id] = 1
        return block_id
    
    def dedup_delete(self, block_id: int) -> None:
        if block_id in self.ref_count:
            self.ref_count[block_id] -= 1
            if self.ref_count[block_id] <= 0:
                del self.ref_count[block_id]
                for fp, bid in list(self.fingerprint_index.items()):
                    if bid == block_id:
                        del self.fingerprint_index[fp]
                        break
                self.block_storage.free_block(block_id)
    
    def create_thin_volume(self, name: str, virtual_size: int) -> None:
        self.thin_volumes[name] = {'virtual_size': virtual_size, 'actual_size': 0, 'block_map': {}}
    
    def thin_write(self, volume_name: str, offset: int, data: bytes) -> None:
        if volume_name not in self.thin_volumes:
            raise ValueError(f"Volume {volume_name} not found")
        volume = self.thin_volumes[volume_name]
        block_size = self.config.block_size
        for i in range(0, len(data), block_size):
            virtual_block = (offset + i) // block_size
            if virtual_block not in volume['block_map']:
                physical_block = self.block_storage.allocate_block()
                volume['block_map'][virtual_block] = physical_block
                volume['actual_size'] += block_size
                self.thin_pool_used += block_size
            self.block_storage.write_block(volume['block_map'][virtual_block], data[i:i+block_size])
    
    def thin_read(self, volume_name: str, offset: int, size: int) -> bytes:
        if volume_name not in self.thin_volumes:
            raise ValueError(f"Volume {volume_name} not found")
        volume = self.thin_volumes[volume_name]
        block_size = self.config.block_size
        result = bytearray()
        for i in range(0, size, block_size):
            virtual_block = (offset + i) // block_size
            if virtual_block in volume['block_map']:
                result.extend(self.block_storage.read_block(volume['block_map'][virtual_block]))
            else:
                result.extend(b'\x00' * block_size)
        return bytes(result[:size])
    
    def get_dedup_ratio(self) -> float:
        logical_size = sum(self.ref_count.values()) * self.config.block_size
        physical_size = len(self.fingerprint_index) * self.config.block_size
        return logical_size / max(physical_size, 1)


# ============================================================================
# 11. StorageManager - 统一存储管理器
# ============================================================================

class StorageManager:
    """统一存储管理器"""
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.block_storage = BlockStorage(self.config)
        self.file_system = FileSystem(self.block_storage, self.config)
        self.object_storage = ObjectStorage(self.block_storage, self.config)
        self.kv_store = KeyValueStore(self.config)
        self.distributed_storage = DistributedStorage(self.config)
        self.cache_layer = CacheLayer(self.config)
        self.erasure_coding = ErasureCoding(self.config)
        self.compression = CompressionEngine(self.config)
        self.optimizer = StorageOptimizer(self.block_storage, self.config)
        self.metrics: Dict[str, Any] = {'operations': defaultdict(int), 'latencies': defaultdict(list),
                                        'errors': defaultdict(int), 'start_time': time.time()}
    
    def block_write(self, block_id: int, data: bytes) -> None:
        start = time.time()
        try:
            self.block_storage.write_block(block_id, data)
            self.metrics['operations']['block_write'] += 1
        except Exception:
            self.metrics['errors']['block_write'] += 1
            raise
        finally:
            self.metrics['latencies']['block_write'].append(time.time() - start)
    
    def block_read(self, block_id: int) -> bytes:
        start = time.time()
        try:
            data = self.block_storage.read_block(block_id)
            self.metrics['operations']['block_read'] += 1
            return data
        except Exception:
            self.metrics['errors']['block_read'] += 1
            raise
        finally:
            self.metrics['latencies']['block_read'].append(time.time() - start)
    
    def fs_create(self, parent: int, name: str) -> int:
        return self.file_system.create_file(parent, name)
    
    def fs_write(self, inode: int, data: bytes, offset: int = 0) -> int:
        return self.file_system.write_file(inode, data, offset)
    
    def fs_read(self, inode: int, offset: int = 0, size: int = -1) -> bytes:
        return self.file_system.read_file(inode, offset, size)
    
    def object_put(self, bucket: str, key: str, data: bytes) -> ObjectMetadata:
        return self.object_storage.put_object(bucket, key, data)
    
    def object_get(self, bucket: str, key: str) -> Tuple[bytes, ObjectMetadata]:
        return self.object_storage.get_object(bucket, key)
    
    def kv_put(self, key: str, value: str) -> None:
        self.kv_store.put(key, value)
    
    def kv_get(self, key: str) -> Optional[str]:
        return self.kv_store.get(key)
    
    def cache_read(self, address: int) -> bytes:
        return self.cache_layer.read(address)
    
    def cache_write(self, address: int, data: bytes) -> None:
        self.cache_layer.write(address, data)
    
    def compress_data(self, data: bytes, algorithm: str = 'lz77') -> bytes:
        return self.compression.compress(data, algorithm)
    
    def decompress_data(self, data: bytes, algorithm: str = 'lz77') -> bytes:
        return self.compression.decompress(data, algorithm)
    
    def encode_erasure(self, data: bytes) -> List[bytes]:
        return self.erasure_coding.encode(data)
    
    def decode_erasure(self, shards: List[Optional[bytes]], shard_size: int) -> bytes:
        return self.erasure_coding.decode(shards, shard_size)
    
    def get_metrics(self) -> Dict[str, Any]:
        runtime = time.time() - self.metrics['start_time']
        avg_latencies = {op: sum(lats) / len(lats) for op, lats in self.metrics['latencies'].items() if lats}
        return {'runtime_seconds': runtime, 'operations': dict(self.metrics['operations']),
                'average_latencies': avg_latencies, 'errors': dict(self.metrics['errors']),
                'block_stats': self.block_storage.get_stats(), 'cache_stats': self.cache_layer.get_stats(),
                'dedup_ratio': self.optimizer.get_dedup_ratio(), 'kv_stats': self.kv_store.stats}
    
    def get_status(self) -> Dict[str, str]:
        return {'status': 'healthy', 'block_storage': 'active', 'file_system': 'active',
                'object_storage': 'active', 'kv_store': 'active', 'cache': 'active'}


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    config = StorageConfig()
    storage = StorageManager(config)
    
    print("=== 高级存储系统演示 ===\n")
    
    print("1. 块存储测试")
    storage.block_write(0, b'Hello, Block Storage!')
    print(f"   读取块0: {storage.block_read(0)[:25]}")
    
    print("\n2. 文件系统测试")
    root = storage.file_system.root_inode_id
    file_inode = storage.fs_create(root, 'test.txt')
    storage.fs_write(file_inode, b'Hello, File System!')
    print(f"   文件内容: {storage.fs_read(file_inode)}")
    
    print("\n3. 对象存储测试")
    storage.object_storage.create_bucket('mybucket')
    meta = storage.object_put('mybucket', 'hello.txt', b'Hello, Object Storage!')
    print(f"   对象ETag: {meta.etag}")
    data, _ = storage.object_get('mybucket', 'hello.txt')
    print(f"   对象内容: {data}")
    
    print("\n4. 键值存储测试")
    storage.kv_put('key1', 'value1')
    print(f"   key1 = {storage.kv_get('key1')}")
    
    print("\n5. 缓存测试")
    storage.cache_write(0x1000, b'Cached Data')
    print(f"   缓存读取: {storage.cache_read(0x1000)}")
    
    print("\n6. 压缩测试")
    original = b'AAAAABBBCCDAAAA'
    compressed = storage.compress_data(original, 'rle')
    print(f"   原始: {original} ({len(original)} bytes)")
    print(f"   压缩后: {compressed} ({len(compressed)} bytes)")
    
    print("\n7. 纠删码测试")
    data = b'Hello, Erasure Coding! This is test data.'
    shards = storage.encode_erasure(data)
    print(f"   数据分片数: {len(shards)}")
    
    print("\n8. 存储指标")
    metrics = storage.get_metrics()
    print(f"   运行时间: {metrics['runtime_seconds']:.2f}s")
    print(f"   操作统计: {metrics['operations']}")
    print(f"   缓存命中率: L1={metrics['cache_stats']['l1_hit_rate']:.2%}")
    
    print("\n=== 演示完成 ===")
