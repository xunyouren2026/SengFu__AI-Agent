"""
AGI Unified Framework - 数据库工具函数模块

本模块提供数据库操作中常用的工具函数，包括：
- 密码哈希和验证（bcrypt）
- API 密钥加密/解密（Fernet）
- JSON 序列化/反序列化
- 分页工具
- 时间格式化
- 数据验证
- 字符串处理
- 文件工具
- 通用辅助函数

设计原则:
    1. 所有加密依赖可选，优雅降级
    2. 完整的类型注解
    3. 详细的文档字符串
    4. 线程安全

依赖:
    - bcrypt >= 3.2 (可选，密码哈希)
    - cryptography >= 3.4 (可选，密钥加密)
"""

import os
import re
import json
import uuid
import hashlib
import hmac
import base64
import logging
import secrets
import string
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple, Union, TypeVar, Generic

logger = logging.getLogger(__name__)

# 尝试导入加密库
try:
    import bcrypt

    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning(
        "bcrypt 未安装。密码哈希功能将使用降级方案（SHA-256）。"
        "建议安装: pip install bcrypt"
    )

try:
    from cryptography.fernet import Fernet

    FERNET_AVAILABLE = True
except ImportError:
    FERNET_AVAILABLE = False
    Fernet = Any  # 占位符类型，避免NameError
    logger.warning(
        "cryptography 未安装。API 密钥加密功能将不可用。"
        "建议安装: pip install cryptography"
    )

# 全局 Fernet 实例
_fernet_instance: Optional[Fernet] = None
_encryption_key: Optional[bytes] = None


# ============================================================
# 密码哈希和验证
# ============================================================

def hash_password(password: str, rounds: int = 12) -> str:
    """
    使用 bcrypt 对密码进行哈希

    使用 bcrypt 算法对明文密码进行安全哈希。如果 bcrypt 不可用，
    则降级使用 SHA-256 + 盐值方案。

    参数:
        password: 明文密码
        rounds: bcrypt 工作因子（10-12 推荐，数值越大越安全但越慢）

    返回:
        哈希后的密码字符串

    异常:
        ValueError: 当密码为空时抛出

    示例:
        >>> hashed = hash_password("my_secure_password")
        >>> print(hashed)  # $2b$12$...
    """
    if not password:
        raise ValueError("密码不能为空")

    if BCRYPT_AVAILABLE:
        try:
            password_bytes = password.encode("utf-8")
            salt = bcrypt.gensalt(rounds=rounds)
            hashed = bcrypt.hashpw(password_bytes, salt)
            return hashed.decode("utf-8")
        except Exception as e:
            logger.error(f"bcrypt 哈希失败: {e}")
            # 降级到 SHA-256
            return _hash_password_fallback(password)
    else:
        return _hash_password_fallback(password)


def verify_password(password: str, password_hash: str) -> bool:
    """
    验证密码是否匹配哈希

    参数:
        password: 明文密码
        password_hash: 存储的密码哈希

    返回:
        密码是否匹配

    示例:
        >>> hashed = hash_password("my_password")
        >>> verify_password("my_password", hashed)  # True
        >>> verify_password("wrong_password", hashed)  # False
    """
    if not password or not password_hash:
        return False

    if BCRYPT_AVAILABLE:
        try:
            password_bytes = password.encode("utf-8")
            hash_bytes = password_hash.encode("utf-8")
            return bcrypt.checkpw(password_bytes, hash_bytes)
        except Exception as e:
            logger.error(f"bcrypt 验证失败: {e}")
            return _verify_password_fallback(password, password_hash)
    else:
        return _verify_password_fallback(password, password_hash)


def _hash_password_fallback(password: str) -> str:
    """
    密码哈希降级方案（SHA-256 + 盐值）

    当 bcrypt 不可用时使用此方案。安全性低于 bcrypt。

    参数:
        password: 明文密码

    返回:
        格式为 "sha256$salt$hash" 的哈希字符串
    """
    salt = secrets.token_hex(16)
    salted = f"{salt}:{password}".encode("utf-8")
    hash_value = hashlib.sha256(salted).hexdigest()
    return f"sha256${salt}${hash_value}"


def _verify_password_fallback(password: str, password_hash: str) -> bool:
    """
    密码验证降级方案

    参数:
        password: 明文密码
        password_hash: 存储的密码哈希

    返回:
        密码是否匹配
    """
    try:
        parts = password_hash.split("$")
        if len(parts) == 3 and parts[0] == "sha256":
            salt = parts[1]
            stored_hash = parts[2]
            salted = f"{salt}:{password}".encode("utf-8")
            computed_hash = hashlib.sha256(salted).hexdigest()
            return hmac.compare_digest(computed_hash, stored_hash)
    except Exception as e:
        logger.error(f"降级密码验证失败: {e}")
    return False


def generate_password(length: int = 16, include_special: bool = True) -> str:
    """
    生成随机安全密码

    参数:
        length: 密码长度（最小 8）
        include_special: 是否包含特殊字符

    返回:
        随机生成的密码字符串

    示例:
        >>> pwd = generate_password(16)
        >>> print(pwd)  # 例如: "aB3$xK9#mP2&nQ5z"
    """
    length = max(8, length)
    alphabet = string.ascii_letters + string.digits
    if include_special:
        alphabet += "!@#$%^&*()-_=+[]{}|;:,.<>?"

    # 确保至少包含每种字符类型
    password_chars = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
    ]
    if include_special:
        password_chars.append(secrets.choice("!@#$%^&*()-_=+[]{}|;:,.<>?"))

    # 填充剩余长度
    remaining = length - len(password_chars)
    password_chars.extend(secrets.choice(alphabet) for _ in range(remaining))

    # 打乱顺序
    result = list(password_chars)
    secrets.SystemRandom().shuffle(result)
    return "".join(result)


# ============================================================
# API 密钥加密/解密
# ============================================================

def init_encryption(key: Optional[str] = None) -> None:
    """
    初始化加密系统

    使用提供的密钥或自动生成密钥来初始化 Fernet 加密实例。

    参数:
        key: Base64 编码的 32 字节密钥。如果为 None，则从环境变量
             ENCRYPTION_KEY 读取，或自动生成新密钥。

    示例:
        >>> init_encryption()  # 自动生成密钥
        >>> init_encryption("my-base64-key-here")
    """
    global _fernet_instance, _encryption_key

    if key:
        _encryption_key = key.encode("utf-8")
    elif os.environ.get("ENCRYPTION_KEY"):
        _encryption_key = os.environ["ENCRYPTION_KEY"].encode("utf-8")
    else:
        # 自动生成密钥
        _encryption_key = Fernet.generate_key() if FERNET_AVAILABLE else None
        if _encryption_key:
            logger.warning(
                "已自动生成加密密钥。请设置 ENCRYPTION_KEY 环境变量以持久化密钥。"
                "密钥: %s", _encryption_key.decode("utf-8")
            )

    if FERNET_AVAILABLE and _encryption_key:
        try:
            _fernet_instance = Fernet(_encryption_key)
            logger.info("加密系统初始化成功")
        except Exception as e:
            logger.error(f"加密系统初始化失败: {e}")
            _fernet_instance = None
    else:
        logger.warning("加密系统不可用（cryptography 未安装）")


def encrypt_api_key(api_key: str) -> str:
    """
    加密 API 密钥

    使用 Fernet 对称加密算法加密 API 密钥。

    参数:
        api_key: 明文 API 密钥

    返回:
        加密后的密文字符串（Base64 编码）

    异常:
        RuntimeError: 当加密系统未初始化时抛出

    示例:
        >>> init_encryption()
        >>> encrypted = encrypt_api_key("sk-abc123...")
        >>> print(encrypted)  # gAAAAA...
    """
    if not _fernet_instance:
        raise RuntimeError(
            "加密系统未初始化。请先调用 init_encryption()。"
        )

    try:
        key_bytes = api_key.encode("utf-8")
        encrypted = _fernet_instance.encrypt(key_bytes)
        return encrypted.decode("utf-8")
    except Exception as e:
        logger.error(f"API 密钥加密失败: {e}")
        raise


def decrypt_api_key(encrypted_key: str) -> str:
    """
    解密 API 密钥

    使用 Fernet 对称加密算法解密 API 密钥。

    参数:
        encrypted_key: 加密后的密文字符串

    返回:
        明文 API 密钥

    异常:
        RuntimeError: 当加密系统未初始化时抛出

    示例:
        >>> init_encryption()
        >>> decrypted = decrypt_api_key(encrypted)
        >>> print(decrypted)  # sk-abc123...
    """
    if not _fernet_instance:
        raise RuntimeError(
            "加密系统未初始化。请先调用 init_encryption()。"
        )

    try:
        encrypted_bytes = encrypted_key.encode("utf-8")
        decrypted = _fernet_instance.decrypt(encrypted_bytes)
        return decrypted.decode("utf-8")
    except Exception as e:
        logger.error(f"API 密钥解密失败: {e}")
        raise


def generate_api_key(prefix: str = "agi") -> str:
    """
    生成 API 密钥

    生成格式为 {prefix}_{random_string} 的 API 密钥。

    参数:
        prefix: 密钥前缀

    返回:
        生成的 API 密钥字符串

    示例:
        >>> key = generate_api_key("agi")
        >>> print(key)  # agi_k8x2Pm9QnR4sT7vW1yZ3...
    """
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}_{random_part}"


# ============================================================
# JSON 序列化工具
# ============================================================

def serialize_json(data: Any) -> Optional[str]:
    """
    将 Python 对象序列化为 JSON 字符串

    支持自定义类型的序列化，包括 datetime、UUID 等。

    参数:
        data: 要序列化的 Python 对象

    返回:
        JSON 字符串或 None（序列化失败时）

    示例:
        >>> serialize_json({"key": "value", "time": datetime.now()})
        '{"key": "value", "time": "2024-01-01T00:00:00+00:00"}'
    """
    try:
        return json.dumps(data, default=_json_serializer, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON 序列化失败: {e}")
        return None


def deserialize_json(json_str: str) -> Optional[Any]:
    """
    将 JSON 字符串反序列化为 Python 对象

    参数:
        json_str: JSON 字符串

    返回:
        Python 对象或 None（反序列化失败时）

    示例:
        >>> data = deserialize_json('{"key": "value"}')
        >>> print(data)  # {'key': 'value'}
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"JSON 反序列化失败: {e}")
        return None


def _json_serializer(obj: Any) -> Any:
    """
    JSON 自定义序列化器

    处理 datetime、UUID、bytes 等非常规类型的序列化。

    参数:
        obj: 要序列化的对象

    返回:
        可序列化的值

    异常:
        TypeError: 当类型不支持时抛出
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    安全的 JSON 解析（失败返回默认值）

    参数:
        json_str: JSON 字符串
        default: 解析失败时的默认返回值

    返回:
        解析结果或默认值
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def merge_json(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个 JSON 字典

    参数:
        base: 基础字典
        override: 覆盖字典

    返回:
        合并后的新字典

    示例:
        >>> merge_json({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}})
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_json(result[key], value)
        else:
            result[key] = value
    return result


# ============================================================
# 分页工具
# ============================================================

T = TypeVar("T")


class PaginationResult(Generic[T]):
    """
    分页结果封装类

    属性:
        items: 当前页的数据列表
        total: 总记录数
        page: 当前页码（从 1 开始）
        page_size: 每页记录数
        total_pages: 总页数
        has_next: 是否有下一页
        has_prev: 是否有上一页
    """

    def __init__(
        self,
        items: List[T],
        total: int,
        page: int = 1,
        page_size: int = 20,
    ) -> None:
        self.items = items
        self.total = total
        self.page = page
        self.page_size = page_size
        self.total_pages = max(1, (total + page_size - 1) // page_size)
        self.has_next = page < self.total_pages
        self.has_prev = page > 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "items": self.items,
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
        }

    def __repr__(self) -> str:
        return (
            f"<PaginationResult(page={self.page}, total={self.total}, "
            f"total_pages={self.total_pages})>"
        )


def paginate_query(
    query: Any,
    page: int = 1,
    page_size: int = 20,
    max_page_size: int = 100,
) -> Tuple[Any, int, int]:
    """
    对 SQLAlchemy 查询应用分页

    参数:
        query: SQLAlchemy Query 对象
        page: 页码（从 1 开始）
        page_size: 每页记录数
        max_page_size: 最大每页记录数

    返回:
        (分页后的查询, 总记录数, 总页数)

    示例:
        >>> query = db.query(User)
        >>> paginated_query, total, pages = paginate_query(query, page=2, page_size=10)
        >>> users = paginated_query.all()
    """
    page = max(1, page)
    page_size = min(max(1, page_size), max_page_size)

    # 计算总数
    total = query.count()

    # 计算偏移量
    offset = (page - 1) * page_size
    total_pages = max(1, (total + page_size - 1) // page_size)

    # 应用分页
    paginated_query = query.offset(offset).limit(page_size)

    return paginated_query, total, total_pages


def paginate_list(
    items: List[T],
    page: int = 1,
    page_size: int = 20,
) -> PaginationResult[T]:
    """
    对列表进行分页

    参数:
        items: 数据列表
        page: 页码（从 1 开始）
        page_size: 每页记录数

    返回:
        分页结果

    示例:
        >>> result = paginate_list([1, 2, 3, 4, 5], page=1, page_size=2)
        >>> result.items  # [1, 2]
        >>> result.total  # 5
    """
    total = len(items)
    page = max(1, page)
    page_size = max(1, page_size)
    offset = (page - 1) * page_size
    page_items = items[offset:offset + page_size]
    return PaginationResult(page_items, total, page, page_size)


# ============================================================
# 时间工具
# ============================================================

def utc_now() -> datetime:
    """
    获取当前 UTC 时间

    返回:
        带时区信息的当前 UTC 时间

    示例:
        >>> now = utc_now()
        >>> print(now.tzinfo)  # UTC
    """
    return datetime.now(timezone.utc)


def format_datetime(
    dt: Optional[datetime],
    format_str: str = "%Y-%m-%d %H:%M:%S",
    timezone_name: Optional[str] = None,
) -> str:
    """
    格式化日期时间

    参数:
        dt: 日期时间对象
        format_str: 格式化字符串
        timezone_name: 时区名称（如 "Asia/Shanghai"）

    返回:
        格式化后的时间字符串

    示例:
        >>> format_datetime(datetime.now(timezone.utc))
        '2024-01-01 12:00:00'
    """
    if dt is None:
        return ""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    if timezone_name:
        try:
            from zoneinfo import ZoneInfo
            target_tz = ZoneInfo(timezone_name)
            dt = dt.astimezone(target_tz)
        except Exception:
            pass

    return dt.strftime(format_str)


def parse_datetime(
    dt_str: str,
    format_str: Optional[str] = None,
) -> Optional[datetime]:
    """
    解析日期时间字符串

    参数:
        dt_str: 日期时间字符串
        format_str: 格式化字符串（None 则自动检测）

    返回:
        日期时间对象或 None
    """
    if not dt_str:
        return None

    if format_str:
        try:
            return datetime.strptime(dt_str, format_str).replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    # 尝试常见格式
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    return None


def parse_duration(duration_str: str) -> Optional[int]:
    """
    解析持续时间字符串为秒数

    支持格式: "30s", "5m", "2h", "1d", "1h30m", "1d2h3m4s"

    参数:
        duration_str: 持续时间字符串

    返回:
        秒数或 None

    示例:
        >>> parse_duration("1h30m")  # 5400
        >>> parse_duration("2d")  # 172800
    """
    if not duration_str:
        return None

    pattern = r"(?:(\d+)d)?\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?"
    match = re.match(pattern, duration_str.strip().lower())
    if not match:
        return None

    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)

    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total if total > 0 else None


def format_duration(seconds: int) -> str:
    """
    将秒数格式化为人类可读的持续时间字符串

    参数:
        seconds: 秒数

    返回:
        格式化字符串

    示例:
        >>> format_duration(3661)  # "1h 1m 1s"
        >>> format_duration(86400)  # "1d"
    """
    if seconds < 0:
        return "0s"

    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小

    参数:
        size_bytes: 文件大小（字节）

    返回:
        格式化后的字符串

    示例:
        >>> format_file_size(1024)  # "1.00 KB"
        >>> format_file_size(1048576)  # "1.00 MB"
    """
    if size_bytes < 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"


# ============================================================
# 数据验证工具
# ============================================================

def validate_email(email: str) -> bool:
    """
    验证电子邮件地址格式

    参数:
        email: 电子邮件地址

    返回:
        是否有效

    示例:
        >>> validate_email("user@example.com")  # True
        >>> validate_email("invalid-email")  # False
    """
    if not email:
        return False
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_username(username: str, min_length: int = 3, max_length: int = 64) -> bool:
    """
    验证用户名格式

    规则:
        - 长度在 min_length 和 max_length 之间
        - 只包含字母、数字、下划线和连字符
        - 以字母或数字开头

    参数:
        username: 用户名
        min_length: 最小长度
        max_length: 最大长度

    返回:
        是否有效

    示例:
        >>> validate_username("user_123")  # True
        >>> validate_username("-invalid")  # False
    """
    if not username:
        return False
    if len(username) < min_length or len(username) > max_length:
        return False
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$"
    return bool(re.match(pattern, username))


def validate_password_strength(
    password: str,
    min_length: int = 8,
    require_uppercase: bool = True,
    require_lowercase: bool = True,
    require_digit: bool = True,
    require_special: bool = False,
) -> Tuple[bool, List[str]]:
    """
    验证密码强度

    参数:
        password: 密码
        min_length: 最小长度
        require_uppercase: 是否要求大写字母
        require_lowercase: 是否要求小写字母
        require_digit: 是否要求数字
        require_special: 是否要求特殊字符

    返回:
        (是否有效, 错误消息列表)

    示例:
        >>> is_valid, errors = validate_password_strength("MyP@ss123")
        >>> is_valid  # True
    """
    errors: List[str] = []

    if len(password) < min_length:
        errors.append(f"密码长度不能少于 {min_length} 个字符")

    if require_uppercase and not re.search(r"[A-Z]", password):
        errors.append("密码必须包含至少一个大写字母")

    if require_lowercase and not re.search(r"[a-z]", password):
        errors.append("密码必须包含至少一个小写字母")

    if require_digit and not re.search(r"\d", password):
        errors.append("密码必须包含至少一个数字")

    if require_special and not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("密码必须包含至少一个特殊字符")

    return len(errors) == 0, errors


def validate_url(url: str) -> bool:
    """
    验证 URL 格式

    参数:
        url: URL 字符串

    返回:
        是否有效
    """
    if not url:
        return False
    pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(pattern, url))


# ============================================================
# 字符串处理工具
# ============================================================

def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    清理字符串

    去除首尾空白、控制字符，并可选截断。

    参数:
        text: 输入字符串
        max_length: 最大长度（None 则不截断）

    返回:
        清理后的字符串

    示例:
        >>> sanitize_string("  hello\\x00world  ", max_length=8)
        'hellowor'
    """
    if not text:
        return ""
    # 去除控制字符
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    cleaned = cleaned.strip()
    if max_length is not None and len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本

    参数:
        text: 输入文本
        max_length: 最大长度
        suffix: 截断后缀

    返回:
        截断后的文本

    示例:
        >>> truncate_text("这是一段很长的文本", max_length=5)
        '这是一段很...'
    """
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length - len(suffix)] + suffix


def slugify(text: str) -> str:
    """
    将文本转换为 URL 友好的 slug

    参数:
        text: 输入文本

    返回:
        slug 字符串

    示例:
        >>> slugify("Hello World! 你好世界")
        'hello-world-你好世界'
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """
    遮蔽敏感数据

    参数:
        data: 敏感数据字符串
        visible_chars: 可见字符数

    返回:
        遮蔽后的字符串

    示例:
        >>> mask_sensitive_data("sk-abc123def456")
        'sk-a******************6'
    """
    if not data:
        return ""
    if len(data) <= visible_chars * 2:
        return "*" * len(data)
    prefix = data[:visible_chars]
    suffix = data[-visible_chars:]
    masked_length = len(data) - visible_chars * 2
    return f"{prefix}{'*' * masked_length}{suffix}"


def escape_like(query: str) -> str:
    """
    转义 SQL LIKE 查询中的特殊字符

    参数:
        query: 查询字符串

    返回:
        转义后的字符串
    """
    return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ============================================================
# 文件工具
# ============================================================

def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    计算文件哈希值

    参数:
        file_path: 文件路径
        algorithm: 哈希算法（sha256, sha512, md5）

    返回:
        十六进制哈希字符串

    示例:
        >>> hash_value = compute_file_hash("/path/to/file.txt")
        >>> print(hash_value)  # a1b2c3...
    """
    hash_func = hashlib.new(algorithm)
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except (IOError, OSError) as e:
        logger.error(f"计算文件哈希失败: {e}")
        return ""


def generate_uuid() -> str:
    """
    生成 UUID 字符串

    返回:
        UUID v4 字符串

    示例:
        >>> uuid_str = generate_uuid()
        >>> print(uuid_str)  # 550e8400-e29b-41d4-a716-446655440000
    """
    return str(uuid.uuid4())


def generate_short_id(length: int = 8) -> str:
    """
    生成短随机 ID

    参数:
        length: ID 长度

    返回:
        随机 ID 字符串

    示例:
        >>> short_id = generate_short_id()
        >>> print(short_id)  # aB3xK9mP
    """
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ============================================================
# 通用辅助函数
# ============================================================

def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    将列表分割为指定大小的块

    参数:
        lst: 输入列表
        chunk_size: 每块大小

    返回:
        分割后的列表

    示例:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    if chunk_size <= 0:
        return [lst]
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(
    base: Dict[str, Any],
    *overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    深度合并多个字典

    参数:
        base: 基础字典
        *overrides: 覆盖字典（按顺序应用）

    返回:
        合并后的新字典

    示例:
        >>> merge_dicts({"a": 1}, {"b": 2}, {"c": 3})
        {'a': 1, 'b': 2, 'c': 3}
    """
    result = base.copy()
    for override in overrides:
        result = merge_json(result, override)
    return result


def deep_get(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    深度获取嵌套字典的值

    参数:
        data: 字典
        key_path: 键路径，用点号分隔（如 "a.b.c"）
        default: 默认值

    返回:
        对应的值或默认值

    示例:
        >>> deep_get({"a": {"b": {"c": 42}}}, "a.b.c")
        42
        >>> deep_get({"a": {}}, "a.b.c", "default")
        'default'
    """
    keys = key_path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def deep_set(data: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """
    深度设置嵌套字典的值

    参数:
        data: 字典
        key_path: 键路径，用点号分隔
        value: 要设置的值

    返回:
        修改后的字典

    示例:
        >>> deep_set({"a": {}}, "a.b.c", 42)
        {'a': {'b': {'c': 42}}}
    """
    keys = key_path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return data


def remove_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    移除字典中值为 None 的项

    参数:
        data: 输入字典

    返回:
        清理后的字典

    示例:
        >>> remove_none_values({"a": 1, "b": None, "c": 0})
        {'a': 1, 'c': 0}
    """
    return {k: v for k, v in data.items() if v is not None}


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> Dict[str, Any]:
    """
    扁平化嵌套字典

    参数:
        data: 嵌套字典
        parent_key: 父键前缀
        separator: 键分隔符

    返回:
        扁平化后的字典

    示例:
        >>> flatten_dict({"a": {"b": {"c": 1}}, "d": 2})
        {'a.b.c': 1, 'd': 2}
    """
    items: List[Tuple[str, Any]] = []
    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(
    data: Dict[str, Any],
    separator: str = ".",
) -> Dict[str, Any]:
    """
    将扁平化字典还原为嵌套字典

    参数:
        data: 扁平化字典
        separator: 键分隔符

    返回:
        嵌套字典

    示例:
        >>> unflatten_dict({"a.b.c": 1, "d": 2})
        {'a': {'b': {'c': 1}}, 'd': 2}
    """
    result: Dict[str, Any] = {}
    for key, value in data.items():
        parts = key.split(separator)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def retry_on_failure(
    func: Any,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """
    带重试的函数执行器

    参数:
        func: 要执行的函数
        max_retries: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 退避倍数
        exceptions: 需要重试的异常类型

    返回:
        函数执行结果

    异常:
        最后一次执行的异常

    示例:
        >>> result = retry_on_failure(lambda: db.query(User).all(), max_retries=3)
    """
    last_exception: Optional[Exception] = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"函数执行失败（第 {attempt + 1}/{max_retries} 次），"
                    f"{current_delay:.1f}秒后重试: {e}"
                )
                time.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"函数执行失败，已达最大重试次数: {e}")

    if last_exception:
        raise last_exception
    return None


def measure_time(func_name: Optional[str] = None) -> Any:
    """
    函数执行时间测量装饰器

    参数:
        func_name: 自定义函数名称（用于日志）

    示例:
        >>> @measure_time("数据库查询")
        ... def query_users():
        ...     return db.query(User).all()
    """
    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = func_name or func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"[{name}] 执行完成: {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                logger.error(f"[{name}] 执行失败: {elapsed:.2f}ms - {e}")
                raise
        return wrapper
    return decorator
