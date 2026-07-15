"""
EXIF 元数据编辑器
支持读取、写入、删除 EXIF 信息
"""

import struct
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum


class ExifTag(Enum):
    """常用 EXIF 标签"""
    # TIFF 基硎标签
    IMAGE_WIDTH = 256
    IMAGE_LENGTH = 257
    BITS_PER_SAMPLE = 258
    COMPRESSION = 259
    PHOTOMETRIC_INTERPRETATION = 262
    MAKE = 271
    MODEL = 272
    STRIP_OFFSETS = 273
    ORIENTATION = 274
    SAMPLES_PER_PIXEL = 277
    ROWS_PER_STRIP = 278
    STRIP_BYTE_COUNTS = 279
    X_RESOLUTION = 282
    Y_RESOLUTION = 283
    PLANAR_CONFIGURATION = 284
    RESOLUTION_UNIT = 296
    DATE_TIME = 306
    ARTIST = 315
    WHITE_POINT = 318
    PRIMARY_CHROMATICITIES = 319
    
    # Exif IFD 标签
    EXIF_IFD_POINTER = 34665
    GPS_IFD_POINTER = 34853
    
    # Exif 特有标签
    EXPOSURE_TIME = 33434
    F_NUMBER = 33437
    EXPOSURE_PROGRAM = 34850
    SPECTRAL_SENSITIVITY = 34852
    ISO_SPEED_RATINGS = 34855
    OECF = 34856
    EXIF_VERSION = 36864
    DATE_TIME_ORIGINAL = 36867
    DATE_TIME_DIGITIZED = 36868
    COMPONENTS_CONFIGURATION = 37121
    COMPRESSED_BITS_PER_PIXEL = 37122
    SHUTTER_SPEED_VALUE = 37377
    APERTURE_VALUE = 37378
    BRIGHTNESS_VALUE = 37379
    EXPOSURE_BIAS_VALUE = 37380
    MAX_APERTURE_VALUE = 37381
    METERING_MODE = 37383
    LIGHT_SOURCE = 37384
    FLASH = 37385
    FOCAL_LENGTH = 37386
    SUBJECT_AREA = 37396
    MAKER_NOTE = 37500
    USER_COMMENT = 37510
    SUB_SEC_TIME = 37520
    SUB_SEC_TIME_ORIGINAL = 37521
    SUB_SEC_TIME_DIGITIZED = 37522
    FLASH_PIX_VERSION = 40960
    COLOR_SPACE = 40961
    PIXEL_X_DIMENSION = 40962
    PIXEL_Y_DIMENSION = 40963
    RELATED_SOUND_FILE = 40964
    EXPOSURE_MODE = 41983
    WHITE_BALANCE = 41986
    DIGITAL_ZOOM_RATIO = 41987
    FOCAL_LENGTH_IN_35MM_FILM = 41989
    SCENE_CAPTURE_TYPE = 41990
    GAIN_CONTROL = 41991
    CONTRAST = 41992
    SATURATION = 41993
    SHARPNESS = 41994
    SUBJECT_DISTANCE_RANGE = 41996
    
    # GPS 标签
    GPS_VERSION_ID = 0
    GPS_LATITUDE_REF = 1
    GPS_LATITUDE = 2
    GPS_LONGITUDE_REF = 3
    GPS_LONGITUDE = 4
    GPS_ALTITUDE_REF = 5
    GPS_ALTITUDE = 6
    GPS_TIME_STAMP = 7
    GPS_SATELLITES = 8
    GPS_STATUS = 9
    GPS_MEASURE_MODE = 10
    GPS_DOP = 11
    GPS_SPEED_REF = 12
    GPS_SPEED = 13
    GPS_TRACK_REF = 14
    GPS_TRACK = 15
    GPS_IMG_DIRECTION_REF = 16
    GPS_IMG_DIRECTION = 17
    GPS_MAP_DATUM = 18
    GPS_DEST_LATITUDE_REF = 19
    GPS_DEST_LATITUDE = 20
    GPS_DEST_LONGITUDE_REF = 21
    GPS_DEST_LONGITUDE = 22
    GPS_DEST_BEARING_REF = 23
    GPS_DEST_BEARING = 24
    GPS_DEST_DISTANCE_REF = 25
    GPS_DEST_DISTANCE = 26
    GPS_PROCESSING_METHOD = 27
    GPS_AREA_INFORMATION = 28
    GPS_DATE_STAMP = 29
    GPS_DIFFERENTIAL = 30


class ExifDataType(Enum):
    """EXIF 数据类型"""
    BYTE = 1
    ASCII = 2
    SHORT = 3
    LONG = 4
    RATIONAL = 5
    SBYTE = 6
    UNDEFINED = 7
    SSHORT = 8
    SLONG = 9
    SRATIONAL = 10


class ExifValue:
    """EXIF 值包装类"""
    
    def __init__(
        self,
        tag: int,
        data_type: ExifDataType,
        value: Any,
        count: int = 1
    ):
        self.tag = tag
        self.data_type = data_type
        self.value = value
        self.count = count
    
    def __repr__(self) -> str:
        return f"ExifValue(tag={self.tag}, type={self.data_type.name}, value={self.value})"
    
    def to_string(self) -> str:
        """转换为字符串表示"""
        if self.data_type == ExifDataType.ASCII:
            if isinstance(self.value, bytes):
                return self.value.rstrip(b'\x00').decode('ascii', errors='replace')
            return str(self.value)
        elif self.data_type == ExifDataType.RATIONAL:
            if isinstance(self.value, tuple):
                return f"{self.value[0]}/{self.value[1]}"
            return str(self.value)
        return str(self.value)


class IFDEntry:
    """IFD 条目"""
    
    def __init__(
        self,
        tag: int,
        data_type: int,
        count: int,
        value_offset: int,
        value: Any = None
    ):
        self.tag = tag
        self.data_type = data_type
        self.count = count
        self.value_offset = value_offset
        self.value = value
    
    def __repr__(self) -> str:
        return (
            f"IFDEntry(tag={self.tag}, type={self.data_type}, "
            f"count={self.count}, offset={self.value_offset})"
        )


class ExifEditor:
    """EXIF 元数据编辑器主类"""
    
    TYPE_SIZES = {
        1: 1,   # BYTE
        2: 1,   # ASCII
        3: 2,   # SHORT
        4: 4,   # LONG
        5: 8,   # RATIONAL
        6: 1,   # SBYTE
        7: 1,   # UNDEFINED
        8: 2,   # SSHORT
        9: 4,   # SLONG
        10: 8,  # SRATIONAL
    }
    
    def __init__(self):
        """初始化 EXIF 编辑器"""
        self._data: bytes = b''
        self._endian: str = '<'  # '<' little-endian, '>' big-endian
        self._ifds: Dict[str, List[IFDEntry]] = {}
        self._exif_offset: int = 0
        self._gps_offset: int = 0
        self._thumbnail_offset: int = 0
        self._thumbnail_length: int = 0
        self._is_modified: bool = False
    
    def load(self, file_path: Union[str, Path]) -> bool:
        """
        从文件加载 EXIF 数据
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            是否加载成功
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        with open(path, 'rb') as f:
            header = f.read(2)
            
            if header == b'\xff\xd8':
                # JPEG 文件
                return self._load_from_jpeg(path)
            elif header == b'II' or header == b'MM':
                # TIFF 文件
                return self._load_from_tiff(path)
            elif header == b'\x89':
                # PNG 文件（可能包含 EXIF）
                return self._load_from_png(path)
            else:
                raise ValueError(f"不支持的文件格式: {path}")
    
    def _load_from_jpeg(self, path: Path) -> bool:
        """从 JPEG 文件加载 EXIF"""
        with open(path, 'rb') as f:
            data = f.read()
        
        # 查找 APP1 标记 (0xFFE1)
        idx = 2
        while idx < len(data) - 4:
            if data[idx] != 0xFF:
                idx += 1
                continue
            
            marker = data[idx + 1]
            
            if marker == 0xE1:
                # APP1 段
                length = struct.unpack('>H', data[idx + 2:idx + 4])[0]
                segment = data[idx + 4:idx + 2 + length]
                
                if segment[:6] == b'Exif\x00\x00':
                    self._data = segment[6:]
                    self._parse_tiff_header()
                    return True
            
            if marker == 0xDA:  # SOS, 图像数据开始
                break
            
            if marker in (0xD8, 0xD9):
                idx += 2
            else:
                length = struct.unpack('>H', data[idx + 2:idx + 4])[0]
                idx += 2 + length
        
        return False
    
    def _load_from_tiff(self, path: Path) -> bool:
        """从 TIFF 文件加载 EXIF"""
        with open(path, 'rb') as f:
            self._data = f.read()
        
        return self._parse_tiff_header()
    
    def _load_from_png(self, path: Path) -> bool:
        """从 PNG 文件加载 EXIF（eXIf 块）"""
        with open(path, 'rb') as f:
            data = f.read()
        
        # 查找 eXIf 块
        idx = 8
        while idx < len(data) - 8:
            chunk_len = struct.unpack('>I', data[idx:idx + 4])[0]
            chunk_type = data[idx + 4:idx + 8]
            
            if chunk_type == b'eXIf':
                self._data = data[idx + 8:idx + 8 + chunk_len]
                return self._parse_tiff_header()
            
            if chunk_type == b'IEND':
                break
            
            idx += 12 + chunk_len
        
        return False
    
    def _parse_tiff_header(self) -> bool:
        """解析 TIFF 头"""
        if len(self._data) < 8:
            return False
        
        # 字节序
        if self._data[:2] == b'II':
            self._endian = '<'
        elif self._data[:2] == b'MM':
            self._endian = '>'
        else:
            return False
        
        # 验证 TIFF 魔数
        magic = struct.unpack(f'{self._endian}H', self._data[2:4])[0]
        if magic != 42:
            return False
        
        # 解析 IFD
        ifd0_offset = struct.unpack(f'{self._endian}I', self._data[4:8])[0]
        self._parse_ifd('IFD0', ifd0_offset)
        
        return True
    
    def _parse_ifd(self, name: str, offset: int) -> None:
        """解析 IFD"""
        if offset >= len(self._data) or offset < 8:
            return
        
        entries = []
        entry_count = struct.unpack(
            f'{self._endian}H',
            self._data[offset:offset + 2]
        )[0]
        
        entry_offset = offset + 2
        
        for i in range(entry_count):
            if entry_offset + 12 > len(self._data):
                break
            
            entry_data = self._data[entry_offset:entry_offset + 12]
            tag = struct.unpack(f'{self._endian}H', entry_data[0:2])[0]
            data_type = struct.unpack(f'{self._endian}H', entry_data[2:4])[0]
            count = struct.unpack(f'{self._endian}I', entry_data[4:8])[0]
            value_offset = struct.unpack(f'{self._endian}I', entry_data[8:12])[0]
            
            # 解析值
            value = self._parse_value(data_type, count, value_offset, entry_offset + 8)
            
            entry = IFDEntry(tag, data_type, count, value_offset, value)
            entries.append(entry)
            
            # 处理子 IFD
            if tag == 34665:  # Exif IFD
                self._exif_offset = value_offset
                self._parse_ifd('Exif', value_offset)
            elif tag == 34853:  # GPS IFD
                self._gps_offset = value_offset
                self._parse_ifd('GPS', value_offset)
            
            entry_offset += 12
        
        self._ifds[name] = entries
        
        # 下一个 IFD
        next_ifd_offset = struct.unpack(
            f'{self._endian}I',
            self._data[entry_offset:entry_offset + 4]
        )[0]
        
        if next_ifd_offset > 0 and name == 'IFD0':
            self._parse_ifd('IFD1', next_ifd_offset)
    
    def _parse_value(
        self,
        data_type: int,
        count: int,
        value_offset: int,
        entry_offset: int
    ) -> Any:
        """解析条目值"""
        type_size = self.TYPE_SIZES.get(data_type, 1)
        total_size = type_size * count
        
        # 值是否内联存储
        if total_size <= 4:
            data = self._data[entry_offset:entry_offset + 4]
        else:
            if value_offset >= len(self._data):
                return None
            data = self._data[value_offset:value_offset + total_size]
        
        if data_type == 1:  # BYTE
            return list(data[:count])
        elif data_type == 2:  # ASCII
            return data[:count].rstrip(b'\x00').decode('ascii', errors='replace')
        elif data_type == 3:  # SHORT
            values = []
            for i in range(count):
                v = struct.unpack(f'{self._endian}H', data[i*2:i*2+2])[0]
                values.append(v)
            return values[0] if count == 1 else values
        elif data_type == 4:  # LONG
            values = []
            for i in range(count):
                v = struct.unpack(f'{self._endian}I', data[i*4:i*4+4])[0]
                values.append(v)
            return values[0] if count == 1 else values
        elif data_type == 5:  # RATIONAL
            values = []
            for i in range(count):
                num = struct.unpack(f'{self._endian}I', data[i*8:i*8+4])[0]
                den = struct.unpack(f'{self._endian}I', data[i*8+4:i*8+8])[0]
                values.append((num, den))
            return values[0] if count == 1 else values
        elif data_type == 7:  # UNDEFINED
            return data[:count]
        elif data_type == 9:  # SLONG
            values = []
            for i in range(count):
                v = struct.unpack(f'{self._endian}i', data[i*4:i*4+4])[0]
                values.append(v)
            return values[0] if count == 1 else values
        elif data_type == 10:  # SRATIONAL
            values = []
            for i in range(count):
                num = struct.unpack(f'{self._endian}i', data[i*8:i*8+4])[0]
                den = struct.unpack(f'{self._endian}i', data[i*8+4:i*8+8])[0]
                values.append((num, den))
            return values[0] if count == 1 else values
        
        return value_offset
    
    def get_tag(self, tag: Union[int, ExifTag]) -> Optional[ExifValue]:
        """
        获取指定标签的值
        
        Args:
            tag: EXIF 标签
            
        Returns:
            ExifValue 对象或 None
        """
        if isinstance(tag, ExifTag):
            tag = tag.value
        
        # 搜索所有 IFD
        for ifd_name, entries in self._ifds.items():
            for entry in entries:
                if entry.tag == tag:
                    data_type = ExifDataType(entry.data_type)
                    return ExifValue(tag, data_type, entry.value, entry.count)
        
        return None
    
    def get_all_tags(self) -> Dict[int, ExifValue]:
        """
        获取所有标签
        
        Returns:
            标签到值的映射
        """
        result = {}
        for ifd_name, entries in self._ifds.items():
            for entry in entries:
                data_type = ExifDataType(entry.data_type)
                result[entry.tag] = ExifValue(
                    entry.tag, data_type, entry.value, entry.count
                )
        return result
    
    def get_common_info(self) -> Dict[str, Any]:
        """
        获取常用 EXIF 信息
        
        Returns:
            常用信息的字典
        """
        info = {}
        
        # 相机信息
        make = self.get_tag(ExifTag.MAKE)
        if make:
            info['make'] = make.to_string()
        
        model = self.get_tag(ExifTag.MODEL)
        if model:
            info['model'] = model.to_string()
        
        # 拍摄时间
        datetime_orig = self.get_tag(ExifTag.DATE_TIME_ORIGINAL)
        if datetime_orig:
            info['datetime_original'] = datetime_orig.to_string()
        
        datetime_mod = self.get_tag(ExifTag.DATE_TIME)
        if datetime_mod:
            info['datetime'] = datetime_mod.to_string()
        
        # 曝光参数
        exposure = self.get_tag(ExifTag.EXPOSURE_TIME)
        if exposure:
            val = exposure.value
            if isinstance(val, tuple):
                info['exposure_time'] = f"1/{val[1]//val[0]}" if val[0] != 0 else "0"
            else:
                info['exposure_time'] = str(val)
        
        f_number = self.get_tag(ExifTag.F_NUMBER)
        if f_number:
            val = f_number.value
            if isinstance(val, tuple) and val[1] != 0:
                info['f_number'] = f"f/{val[0]/val[1]:.1f}"
            else:
                info['f_number'] = str(val)
        
        iso = self.get_tag(ExifTag.ISO_SPEED_RATINGS)
        if iso:
            info['iso'] = iso.value
        
        focal = self.get_tag(ExifTag.FOCAL_LENGTH)
        if focal:
            val = focal.value
            if isinstance(val, tuple) and val[1] != 0:
                info['focal_length'] = f"{val[0]/val[1]:.1f}mm"
            else:
                info['focal_length'] = str(val)
        
        # 图像尺寸
        width = self.get_tag(ExifTag.PIXEL_X_DIMENSION)
        height = self.get_tag(ExifTag.PIXEL_Y_DIMENSION)
        if width and height:
            info['dimensions'] = f"{width.value}x{height.value}"
        
        # GPS 信息
        lat = self.get_tag(ExifTag.GPS_LATITUDE)
        lat_ref = self.get_tag(ExifTag.GPS_LATITUDE_REF)
        lon = self.get_tag(ExifTag.GPS_LONGITUDE)
        lon_ref = self.get_tag(ExifTag.GPS_LONGITUDE_REF)
        
        if lat and lon:
            info['gps'] = self._format_gps(lat.value, lat_ref, lon.value, lon_ref)
        
        return info
    
    def _format_gps(
        self,
        lat: Any,
        lat_ref: Optional[ExifValue],
        lon: Any,
        lon_ref: Optional[ExifValue]
    ) -> str:
        """格式化 GPS 坐标"""
        try:
            # 转换度分秒为十进制度
            def to_decimal(coord: List[Tuple[int, int]]) -> float:
                if len(coord) >= 3:
                    deg = coord[0][0] / coord[0][1] if coord[0][1] != 0 else 0
                    min = coord[1][0] / coord[1][1] if coord[1][1] != 0 else 0
                    sec = coord[2][0] / coord[2][1] if coord[2][1] != 0 else 0
                    return deg + min/60 + sec/3600
                return 0
            
            lat_val = to_decimal(lat)
            lon_val = to_decimal(lon)
            
            if lat_ref and lat_ref.value == 'S':
                lat_val = -lat_val
            if lon_ref and lon_ref.value == 'W':
                lon_val = -lon_val
            
            return f"{lat_val:.6f}, {lon_val:.6f}"
        except Exception:
            return "N/A"
    
    def set_tag(
        self,
        tag: Union[int, ExifTag],
        value: Any,
        data_type: Optional[ExifDataType] = None
    ) -> bool:
        """
        设置标签值
        
        Args:
            tag: EXIF 标签
            value: 值
            data_type: 数据类型（可选）
            
        Returns:
            是否设置成功
        """
        if isinstance(tag, ExifTag):
            tag = tag.value
        
        # 推断数据类型
        if data_type is None:
            if isinstance(value, str):
                data_type = ExifDataType.ASCII
            elif isinstance(value, int):
                data_type = ExifDataType.LONG
            elif isinstance(value, float):
                data_type = ExifDataType.RATIONAL
            elif isinstance(value, bytes):
                data_type = ExifDataType.UNDEFINED
            else:
                data_type = ExifDataType.LONG
        
        # 查找并更新现有条目
        for ifd_name, entries in self._ifds.items():
            for i, entry in enumerate(entries):
                if entry.tag == tag:
                    entry.value = value
                    entry.data_type = data_type.value
                    self._is_modified = True
                    return True
        
        # 创建新条目
        new_entry = IFDEntry(tag, data_type.value, 1, 0, value)
        if 'IFD0' not in self._ifds:
            self._ifds['IFD0'] = []
        self._ifds['IFD0'].append(new_entry)
        self._is_modified = True
        
        return True
    
    def delete_tag(self, tag: Union[int, ExifTag]) -> bool:
        """
        删除标签
        
        Args:
            tag: EXIF 标签
            
        Returns:
            是否删除成功
        """
        if isinstance(tag, ExifTag):
            tag = tag.value
        
        for ifd_name, entries in self._ifds.items():
            for i, entry in enumerate(entries):
                if entry.tag == tag:
                    entries.pop(i)
                    self._is_modified = True
                    return True
        
        return False
    
    def clear_all(self) -> None:
        """清除所有 EXIF 数据"""
        self._ifds = {}
        self._is_modified = True
    
    def clear_gps(self) -> None:
        """清除 GPS 数据"""
        if 'GPS' in self._ifds:
            del self._ifds['GPS']
            self._is_modified = True
        
        # 删除 GPS IFD 指针
        self.delete_tag(ExifTag.GPS_IFD_POINTER)
    
    def save(self, output_path: Union[str, Path]) -> bool:
        """
        保存 EXIF 数据到文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            是否保存成功
        """
        path = Path(output_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        # 构建新的 EXIF 数据
        new_exif = self._build_exif()
        
        with open(path, 'rb') as f:
            header = f.read(2)
        
        if header == b'\xff\xd8':
            return self._save_to_jpeg(path, new_exif)
        elif header == b'II' or header == b'MM':
            return self._save_to_tiff(path, new_exif)
        else:
            raise ValueError(f"不支持的文件格式: {path}")
    
    def _build_exif(self) -> bytes:
        """构建 EXIF 数据"""
        # TIFF 头
        data = bytearray()
        data.extend(b'II')  # Little-endian
        data.extend(struct.pack('<H', 42))  # 魔数
        data.extend(struct.pack('<I', 8))  # IFD0 偏移
        
        # 构建 IFD0
        if 'IFD0' in self._ifds:
            ifd0_data = self._build_ifd(self._ifds['IFD0'])
            data.extend(ifd0_data)
        
        return bytes(data)
    
    def _build_ifd(self, entries: List[IFDEntry]) -> bytes:
        """构建 IFD 数据"""
        data = bytearray()
        
        # 条目数
        data.extend(struct.pack('<H', len(entries)))
        
        # 条目
        for entry in entries:
            data.extend(struct.pack('<H', entry.tag))
            data.extend(struct.pack('<H', entry.data_type))
            data.extend(struct.pack('<I', entry.count))
            
            # 值/偏移（简化处理）
            value_bytes = self._encode_value(entry.value, entry.data_type)
            if len(value_bytes) <= 4:
                data.extend(value_bytes.ljust(4, b'\x00'))
            else:
                # 需要存储偏移，简化处理
                data.extend(b'\x00\x00\x00\x00')
        
        # 下一个 IFD 偏移
        data.extend(struct.pack('<I', 0))
        
        return bytes(data)
    
    def _encode_value(self, value: Any, data_type: int) -> bytes:
        """编码值"""
        if data_type == 2:  # ASCII
            if isinstance(value, str):
                return (value + '\x00').encode('ascii')
            return value
        elif data_type == 3:  # SHORT
            return struct.pack('<H', value)
        elif data_type == 4:  # LONG
            return struct.pack('<I', value)
        elif data_type == 5:  # RATIONAL
            if isinstance(value, tuple):
                return struct.pack('<II', value[0], value[1])
            return struct.pack('<II', value, 1)
        
        return b'\x00\x00\x00\x00'
    
    def _save_to_jpeg(self, path: Path, exif_data: bytes) -> bool:
        """保存到 JPEG 文件"""
        with open(path, 'rb') as f:
            data = bytearray(f.read())
        
        # 查找并替换 APP1 段
        idx = 2
        while idx < len(data) - 4:
            if data[idx] != 0xFF:
                idx += 1
                continue
            
            marker = data[idx + 1]
            
            if marker == 0xE1:
                # 替换现有 APP1
                old_length = struct.unpack('>H', data[idx + 2:idx + 4])[0]
                
                # 构建新的 APP1 段
                new_app1 = bytearray()
                new_app1.extend(b'Exif\x00\x00')
                new_app1.extend(exif_data)
                
                new_length = len(new_app1) + 2  # 包含长度字段本身
                
                # 替换数据
                data[idx + 2:idx + 2 + old_length] = struct.pack('>H', new_length) + new_app1
                break
            
            if marker == 0xDA:
                # 在 SOS 前插入新的 APP1
                new_app1 = bytearray()
                new_app1.extend(b'\xFF\xE1')
                app1_data = b'Exif\x00\x00' + exif_data
                new_app1.extend(struct.pack('>H', len(app1_data) + 2))
                new_app1.extend(app1_data)
                
                data[idx:idx] = new_app1
                break
            
            if marker in (0xD8, 0xD9):
                idx += 2
            else:
                length = struct.unpack('>H', data[idx + 2:idx + 4])[0]
                idx += 2 + length
        
        with open(path, 'wb') as f:
            f.write(data)
        
        return True
    
    def _save_to_tiff(self, path: Path, exif_data: bytes) -> bool:
        """保存到 TIFF 文件"""
        with open(path, 'wb') as f:
            f.write(exif_data)
        return True
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        导出为字典格式
        
        Returns:
            EXIF 数据字典
        """
        result = {}
        
        for ifd_name, entries in self._ifds.items():
            ifd_data = {}
            for entry in entries:
                tag_name = self._get_tag_name(entry.tag)
                ifd_data[tag_name] = {
                    'tag': entry.tag,
                    'type': entry.data_type,
                    'count': entry.count,
                    'value': entry.value
                }
            result[ifd_name] = ifd_data
        
        return result
    
    def _get_tag_name(self, tag: int) -> str:
        """获取标签名称"""
        for name, member in ExifTag.__members__.items():
            if member.value == tag:
                return name
        return f"Tag_{tag}"
    
    def import_from_dict(self, data: Dict[str, Any]) -> bool:
        """
        从字典导入 EXIF 数据
        
        Args:
            data: EXIF 数据字典
            
        Returns:
            是否导入成功
        """
        for ifd_name, ifd_data in data.items():
            entries = []
            for tag_name, tag_data in ifd_data.items():
                entry = IFDEntry(
                    tag=tag_data['tag'],
                    data_type=tag_data['type'],
                    count=tag_data['count'],
                    value_offset=0,
                    value=tag_data['value']
                )
                entries.append(entry)
            self._ifds[ifd_name] = entries
        
        self._is_modified = True
        return True
    
    @property
    def is_modified(self) -> bool:
        """是否已修改"""
        return self._is_modified
    
    @property
    def has_exif(self) -> bool:
        """是否包含 EXIF 数据"""
        return len(self._ifds) > 0
    
    @property
    def has_gps(self) -> bool:
        """是否包含 GPS 数据"""
        return 'GPS' in self._ifds and len(self._ifds['GPS']) > 0
