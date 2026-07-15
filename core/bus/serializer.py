"""
消息序列化模块

提供多种消息序列化方式，支持JSON和二进制序列化，
以及基于Content-Type的自动协商和zlib压缩。
"""

import json
import logging
import struct
import zlib
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """序列化格式枚举"""
    JSON = "application/json"
    MSGPACK = "application/x-msgpack"
    BINARY = "application/octet-stream"


class MessageSerializer:
    """
    消息序列化器

    提供JSON和简易二进制序列化，支持自动格式协商和zlib压缩。

    Usage:
        serializer = MessageSerializer()

        # JSON序列化
        data = serializer.serialize_json({"key": "value"})
        result = serializer.deserialize_json(data)

        # 二进制序列化
        data = serializer.serialize_msgpack({"key": "value"})
        result = serializer.deserialize_msgpack(data)

        # 自动协商
        data, content_type = serializer.auto_negotiate(
            {"key": "value"},
            accept="application/json"
        )
    """

    # 二进制协议类型标记
    _TYPE_NULL = 0x00
    _TYPE_TRUE = 0x01
    _TYPE_FALSE = 0x02
    _TYPE_INT32 = 0x03
    _TYPE_INT64 = 0x04
    _TYPE_FLOAT64 = 0x05
    _TYPE_STRING = 0x06
    _TYPE_LIST = 0x07
    _TYPE_DICT = 0x08
    _TYPE_BYTES = 0x09

    # 压缩标记
    _COMPRESSION_NONE = 0x00
    _COMPRESSION_ZLIB = 0x01

    # 压缩阈值（字节），超过此大小自动压缩
    DEFAULT_COMPRESSION_THRESHOLD = 1024

    def __init__(
        self,
        compression_threshold: int = DEFAULT_COMPRESSION_THRESHOLD,
        default_format: SerializationFormat = SerializationFormat.JSON,
        encoding: str = "utf-8",
    ):
        """
        初始化序列化器

        Args:
            compression_threshold: 压缩阈值（字节）
            default_format: 默认序列化格式
            encoding: 字符串编码
        """
        self.compression_threshold = compression_threshold
        self.default_format = default_format
        self.encoding = encoding

    # ==================== JSON 序列化 ====================

    def serialize_json(
        self,
        data: Any,
        compress: bool = False,
    ) -> bytes:
        """
        JSON序列化

        将数据序列化为JSON格式的字节流。

        Args:
            data: 要序列化的数据
            compress: 是否使用zlib压缩

        Returns:
            序列化后的字节流
        """
        try:
            json_str = json.dumps(data, ensure_ascii=False, default=str)
            raw_bytes = json_str.encode(self.encoding)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"JSON序列化失败: {exc}") from exc

        if compress or (
            self.compression_threshold > 0
            and len(raw_bytes) >= self.compression_threshold
        ):
            compressed = zlib.compress(raw_bytes)
            # 格式: [1字节压缩标记] + [4字节原始长度] + [压缩数据]
            header = struct.pack(
                ">BI",
                self._COMPRESSION_ZLIB,
                len(raw_bytes),
            )
            return header + compressed

        # 格式: [1字节无压缩标记] + [JSON数据]
        return struct.pack(">B", self._COMPRESSION_NONE) + raw_bytes

    def deserialize_json(self, data: bytes) -> Any:
        """
        JSON反序列化

        Args:
            data: 序列化的字节流

        Returns:
            反序列化的数据

        Raises:
            ValueError: 数据格式无效
        """
        if len(data) < 1:
            raise ValueError("数据为空")

        compression_flag = struct.unpack(">B", data[:1])[0]

        if compression_flag == self._COMPRESSION_ZLIB:
            if len(data) < 5:
                raise ValueError("压缩数据格式不完整")
            original_length = struct.unpack(">I", data[1:5])[0]
            compressed_data = data[5:]
            try:
                raw_bytes = zlib.decompress(compressed_data)
            except zlib.error as exc:
                raise ValueError(f"zlib解压失败: {exc}") from exc
        elif compression_flag == self._COMPRESSION_NONE:
            raw_bytes = data[1:]
        else:
            raise ValueError(f"未知的压缩标记: {compression_flag}")

        try:
            json_str = raw_bytes.decode(self.encoding)
            return json.loads(json_str)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"JSON反序列化失败: {exc}") from exc

    # ==================== 二进制序列化 ====================

    def serialize_msgpack(
        self,
        data: Any,
        compress: bool = False,
    ) -> bytes:
        """
        简易MsgPack-like二进制序列化

        使用struct模块实现紧凑的二进制序列化格式。

        格式规范:
        - 每个值以1字节类型标记开头
        - NULL: 0x00
        - TRUE: 0x01
        - FALSE: 0x02
        - INT32: 0x03 + 4字节有符号整数
        - INT64: 0x04 + 8字节有符号整数
        - FLOAT64: 0x05 + 8字节IEEE754双精度
        - STRING: 0x06 + 4字节长度 + UTF-8字节
        - LIST: 0x07 + 4字节元素数量 + 元素序列
        - DICT: 0x08 + 4字节键值对数量 + 键值对序列
        - BYTES: 0x09 + 4字节长度 + 原始字节

        Args:
            data: 要序列化的数据
            compress: 是否使用zlib压缩

        Returns:
            序列化后的字节流
        """
        raw_bytes = self._pack_value(data)

        if compress or (
            self.compression_threshold > 0
            and len(raw_bytes) >= self.compression_threshold
        ):
            compressed = zlib.compress(raw_bytes)
            header = struct.pack(
                ">BI",
                self._COMPRESSION_ZLIB,
                len(raw_bytes),
            )
            return header + compressed

        return struct.pack(">B", self._COMPRESSION_NONE) + raw_bytes

    def deserialize_msgpack(self, data: bytes) -> Any:
        """
        二进制反序列化

        Args:
            data: 序列化的字节流

        Returns:
            反序列化的数据
        """
        if len(data) < 1:
            raise ValueError("数据为空")

        compression_flag = struct.unpack(">B", data[:1])[0]

        if compression_flag == self._COMPRESSION_ZLIB:
            if len(data) < 5:
                raise ValueError("压缩数据格式不完整")
            compressed_data = data[5:]
            try:
                raw_bytes = zlib.decompress(compressed_data)
            except zlib.error as exc:
                raise ValueError(f"zlib解压失败: {exc}") from exc
            offset = 0
            result, _ = self._unpack_value(raw_bytes, offset)
            return result
        elif compression_flag == self._COMPRESSION_NONE:
            raw_bytes = data[1:]
            offset = 0
            result, _ = self._unpack_value(raw_bytes, offset)
            return result
        else:
            raise ValueError(f"未知的压缩标记: {compression_flag}")

    def _pack_value(self, value: Any) -> bytes:
        """将单个值打包为二进制"""
        if value is None:
            return struct.pack(">B", self._TYPE_NULL)

        elif value is True:
            return struct.pack(">B", self._TYPE_TRUE)

        elif value is False:
            return struct.pack(">B", self._TYPE_FALSE)

        elif isinstance(value, bool):
            return struct.pack(
                ">B", self._TYPE_TRUE if value else self._TYPE_FALSE
            )

        elif isinstance(value, int):
            # 尝试使用int32，溢出则使用int64
            try:
                struct.pack(">i", value)
                return struct.pack(">Bi", self._TYPE_INT32, value)
            except struct.error:
                return struct.pack(">Bq", self._TYPE_INT64, value)

        elif isinstance(value, float):
            return struct.pack(">Bd", self._TYPE_FLOAT64, value)

        elif isinstance(value, str):
            encoded = value.encode(self.encoding)
            return (
                struct.pack(">BI", self._TYPE_STRING, len(encoded))
                + encoded
            )

        elif isinstance(value, bytes):
            return (
                struct.pack(">BI", self._TYPE_BYTES, len(value))
                + value
            )

        elif isinstance(value, (list, tuple)):
            parts = [struct.pack(">BI", self._TYPE_LIST, len(value))]
            for item in value:
                parts.append(self._pack_value(item))
            return b"".join(parts)

        elif isinstance(value, dict):
            parts = [struct.pack(">BI", self._TYPE_DICT, len(value))]
            for k, v in value.items():
                key_bytes = k.encode(self.encoding)
                parts.append(
                    struct.pack(">BI", self._TYPE_STRING, len(key_bytes))
                    + key_bytes
                )
                parts.append(self._pack_value(v))
            return b"".join(parts)

        else:
            # 回退到字符串表示
            str_val = str(value)
            encoded = str_val.encode(self.encoding)
            return (
                struct.pack(">BI", self._TYPE_STRING, len(encoded))
                + encoded
            )

    def _unpack_value(self, data: bytes, offset: int) -> Tuple[Any, int]:
        """从二进制数据中解包单个值"""
        if offset >= len(data):
            raise ValueError("数据不完整: 意外的结束")

        type_tag = struct.unpack(">B", data[offset:offset + 1])[0]
        offset += 1

        if type_tag == self._TYPE_NULL:
            return None, offset

        elif type_tag == self._TYPE_TRUE:
            return True, offset

        elif type_tag == self._TYPE_FALSE:
            return False, offset

        elif type_tag == self._TYPE_INT32:
            if offset + 4 > len(data):
                raise ValueError("数据不完整: INT32")
            value = struct.unpack(">i", data[offset:offset + 4])[0]
            return value, offset + 4

        elif type_tag == self._TYPE_INT64:
            if offset + 8 > len(data):
                raise ValueError("数据不完整: INT64")
            value = struct.unpack(">q", data[offset:offset + 8])[0]
            return value, offset + 8

        elif type_tag == self._TYPE_FLOAT64:
            if offset + 8 > len(data):
                raise ValueError("数据不完整: FLOAT64")
            value = struct.unpack(">d", data[offset:offset + 8])[0]
            return value, offset + 8

        elif type_tag == self._TYPE_STRING:
            if offset + 4 > len(data):
                raise ValueError("数据不完整: STRING长度")
            str_len = struct.unpack(">I", data[offset:offset + 4])[0]
            offset += 4
            if offset + str_len > len(data):
                raise ValueError("数据不完整: STRING内容")
            value = data[offset:offset + str_len].decode(self.encoding)
            return value, offset + str_len

        elif type_tag == self._TYPE_BYTES:
            if offset + 4 > len(data):
                raise ValueError("数据不完整: BYTES长度")
            byte_len = struct.unpack(">I", data[offset:offset + 4])[0]
            offset += 4
            if offset + byte_len > len(data):
                raise ValueError("数据不完整: BYTES内容")
            value = data[offset:offset + byte_len]
            return value, offset + byte_len

        elif type_tag == self._TYPE_LIST:
            if offset + 4 > len(data):
                raise ValueError("数据不完整: LIST长度")
            count = struct.unpack(">I", data[offset:offset + 4])[0]
            offset += 4
            result = []
            for _ in range(count):
                item, offset = self._unpack_value(data, offset)
                result.append(item)
            return result, offset

        elif type_tag == self._TYPE_DICT:
            if offset + 4 > len(data):
                raise ValueError("数据不完整: DICT长度")
            count = struct.unpack(">I", data[offset:offset + 4])[0]
            offset += 4
            result = {}
            for _ in range(count):
                key, offset = self._unpack_value(data, offset)
                val, offset = self._unpack_value(data, offset)
                result[key] = val
            return result, offset

        else:
            raise ValueError(f"未知的类型标记: 0x{type_tag:02x}")

    # ==================== 自动协商 ====================

    def auto_negotiate(
        self,
        data: Any,
        accept: Optional[str] = None,
        compress: Optional[bool] = None,
    ) -> Tuple[bytes, str]:
        """
        根据Content-Type自动选择序列化方式

        Args:
            data: 要序列化的数据
            accept: 客户端接受的Content-Type
            compress: 是否压缩（None则根据大小自动判断）

        Returns:
            (序列化后的字节流, Content-Type) 元组
        """
        format_type = self._negotiate_format(accept)

        if format_type == SerializationFormat.JSON:
            should_compress = compress if compress is not None else (
                len(json.dumps(data, ensure_ascii=False, default=str).encode(self.encoding))
                >= self.compression_threshold
            )
            serialized = self.serialize_json(data, compress=should_compress)
            content_type = SerializationFormat.JSON.value
        elif format_type == SerializationFormat.MSGPACK:
            should_compress = compress if compress is not None else False
            serialized = self.serialize_msgpack(data, compress=should_compress)
            content_type = SerializationFormat.MSGPACK.value
        else:
            # 回退到JSON
            serialized = self.serialize_json(data, compress=compress or False)
            content_type = SerializationFormat.JSON.value

        return serialized, content_type

    def auto_deserialize(
        self,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> Any:
        """
        根据Content-Type自动选择反序列化方式

        Args:
            data: 序列化的字节流
            content_type: Content-Type

        Returns:
            反序列化的数据
        """
        if content_type is None:
            content_type = self.default_format.value

        if content_type == SerializationFormat.JSON.value:
            return self.deserialize_json(data)
        elif content_type == SerializationFormat.MSGPACK.value:
            return self.deserialize_msgpack(data)
        else:
            # 尝试JSON
            try:
                return self.deserialize_json(data)
            except ValueError:
                return self.deserialize_msgpack(data)

    def _negotiate_format(
        self, accept: Optional[str]
    ) -> SerializationFormat:
        """
        根据Accept头协商序列化格式

        Args:
            accept: Accept头内容

        Returns:
            序列化格式
        """
        if not accept:
            return self.default_format

        accept_lower = accept.lower()

        if SerializationFormat.MSGPACK.value in accept_lower:
            return SerializationFormat.MSGPACK
        elif SerializationFormat.JSON.value in accept_lower:
            return SerializationFormat.JSON
        elif "application/*" in accept_lower or "*/*" in accept_lower:
            return self.default_format

        return self.default_format

    def get_stats(self) -> Dict[str, Any]:
        """获取序列化器配置信息"""
        return {
            "default_format": self.default_format.value,
            "compression_threshold": self.compression_threshold,
            "encoding": self.encoding,
            "supported_formats": [f.value for f in SerializationFormat],
        }
