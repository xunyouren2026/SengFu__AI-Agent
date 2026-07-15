"""
IoT (Internet of Things) Comprehensive Module
物联网综合模块

包含完整的IoT系统模拟实现:
- IoT设备模拟
- 传感器数据生成
- 网关/桥接
- 流处理
- 时序数据库
- 异常检测
- 数字孪生
- 安全机制
- 能源管理
- 配置管理
- IoT平台
"""

import random
import math
import time
import json
import hashlib
import heapq
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from enum import Enum, auto
from datetime import datetime, timedelta
import threading
import copy


# =============================================================================
# 配置管理 (IoTConfig)
# =============================================================================

@dataclass
class IoTConfig:
    """IoT系统配置管理"""
    # 设备配置
    default_sampling_rate: float = 1.0  # Hz
    default_battery_capacity: float = 1000.0  # mAh
    sleep_power: float = 0.001  # mW
    active_power: float = 10.0  # mW
    transmission_power: float = 50.0  # mW

    # 传感器配置
    noise_std: float = 0.1
    drift_rate: float = 0.001
    calibration_interval: int = 1000  # samples

    # 网关配置
    max_queue_size: int = 10000
    aggregation_window: float = 60.0  # seconds

    # 数据库配置
    retention_days: int = 30
    downsampling_intervals: List[int] = field(default_factory=lambda: [60, 300, 3600])

    # 异常检测配置
    anomaly_threshold: float = 3.0  # Z-score threshold
    window_size: int = 100

    # 安全配置
    token_expiry: int = 3600  # seconds
    max_failed_auth: int = 3

    # 数字孪生配置
    kalman_process_noise: float = 0.01
    kalman_measurement_noise: float = 0.1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IoTConfig':
        """从字典加载配置"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            'default_sampling_rate': self.default_sampling_rate,
            'default_battery_capacity': self.default_battery_capacity,
            'sleep_power': self.sleep_power,
            'active_power': self.active_power,
            'transmission_power': self.transmission_power,
            'noise_std': self.noise_std,
            'drift_rate': self.drift_rate,
            'calibration_interval': self.calibration_interval,
            'max_queue_size': self.max_queue_size,
            'aggregation_window': self.aggregation_window,
            'retention_days': self.retention_days,
            'downsampling_intervals': self.downsampling_intervals,
            'anomaly_threshold': self.anomaly_threshold,
            'window_size': self.window_size,
            'token_expiry': self.token_expiry,
            'max_failed_auth': self.max_failed_auth,
            'kalman_process_noise': self.kalman_process_noise,
            'kalman_measurement_noise': self.kalman_measurement_noise
        }


# =============================================================================
# 传感器 (Sensor)
# =============================================================================

class SensorType(Enum):
    """传感器类型"""
    TEMPERATURE = auto()
    HUMIDITY = auto()
    PRESSURE = auto()
    MOTION = auto()
    LIGHT = auto()


class Sensor:
    """
    传感器数据生成器
    支持高斯噪声模型、漂移、校准和故障注入
    """

    def __init__(self, sensor_type: SensorType, config: IoTConfig):
        self.sensor_type = sensor_type
        self.config = config
        self.drift = 0.0
        self.calibration_offset = 0.0
        self.sample_count = 0
        self.fault_mode = None  # 'stuck', 'spike', 'dropout'
        self.fault_params = {}
        self.last_value = None

        # 传感器参数配置
        self.params = self._get_default_params()

    def _get_default_params(self) -> Dict[str, Any]:
        """获取传感器默认参数"""
        params = {
            SensorType.TEMPERATURE: {'min': -40, 'max': 85, 'unit': '°C', 'base': 25},
            SensorType.HUMIDITY: {'min': 0, 'max': 100, 'unit': '%', 'base': 50},
            SensorType.PRESSURE: {'min': 300, 'max': 1100, 'unit': 'hPa', 'base': 1013},
            SensorType.MOTION: {'min': 0, 'max': 1, 'unit': 'binary', 'base': 0},
            SensorType.LIGHT: {'min': 0, 'max': 100000, 'unit': 'lux', 'base': 500}
        }
        return params.get(self.sensor_type, {'min': 0, 'max': 100, 'unit': 'raw', 'base': 50})

    def read(self, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        读取传感器数据

        Returns:
            包含时间戳、值、单位、质量等信息的字典
        """
        if timestamp is None:
            timestamp = time.time()

        self.sample_count += 1

        # 更新漂移
        self.drift += random.gauss(0, self.config.drift_rate)

        # 定期校准
        if self.sample_count % self.config.calibration_interval == 0:
            self._calibrate()

        # 生成基础值
        base_value = self._generate_base_value(timestamp)

        # 添加噪声
        noisy_value = base_value + random.gauss(0, self.config.noise_std)

        # 应用漂移和校准
        value = noisy_value + self.drift + self.calibration_offset

        # 应用故障模式
        value = self._apply_fault(value)

        # 裁剪到有效范围
        value = max(self.params['min'], min(self.params['max'], value))

        self.last_value = value

        # 计算数据质量 (0-1)
        quality = self._calculate_quality()

        return {
            'timestamp': timestamp,
            'value': round(value, 4),
            'unit': self.params['unit'],
            'sensor_type': self.sensor_type.name,
            'quality': quality,
            'sample_id': self.sample_count
        }

    def _generate_base_value(self, timestamp: float) -> float:
        """生成基础信号值 (可包含周期性变化)"""
        base = self.params['base']

        # 添加周期性变化 (模拟日变化等)
        if self.sensor_type == SensorType.TEMPERATURE:
            # 温度日变化
            hour = (timestamp / 3600) % 24
            variation = 5 * math.sin(2 * math.pi * (hour - 6) / 24)
            return base + variation
        elif self.sensor_type == SensorType.LIGHT:
            # 光照日变化
            hour = (timestamp / 3600) % 24
            if 6 <= hour <= 18:
                variation = self.params['max'] * math.sin(math.pi * (hour - 6) / 12)
                return variation
            return 0
        elif self.sensor_type == SensorType.MOTION:
            # 运动检测 - 随机触发
            return 1 if random.random() < 0.1 else 0

        return base + random.gauss(0, 0.5)

    def _calibrate(self):
        """执行校准"""
        # 简单校准: 将漂移重置为0
        self.calibration_offset = -self.drift
        self.drift = 0.0

    def _apply_fault(self, value: float) -> float:
        """应用故障模式"""
        if self.fault_mode == 'stuck':
            # 卡死故障 - 返回固定值
            return self.fault_params.get('stuck_value', value)
        elif self.fault_mode == 'spike':
            # 尖峰故障
            if random.random() < self.fault_params.get('probability', 0.01):
                return value + random.gauss(0, self.fault_params.get('magnitude', 10))
        elif self.fault_mode == 'dropout':
            # 数据丢失
            if random.random() < self.fault_params.get('probability', 0.05):
                return float('nan')
        return value

    def _calculate_quality(self) -> float:
        """计算数据质量分数"""
        quality = 1.0

        # 漂移影响质量
        quality -= min(0.3, abs(self.drift))

        # 故障模式影响质量
        if self.fault_mode:
            quality -= 0.5

        return max(0.0, min(1.0, quality))

    def inject_fault(self, fault_type: str, **params):
        """注入故障"""
        self.fault_mode = fault_type
        self.fault_params = params

    def clear_fault(self):
        """清除故障"""
        self.fault_mode = None
        self.fault_params = {}

    @staticmethod
    def delta_encode(data: List[float]) -> List[float]:
        """Delta编码压缩"""
        if not data:
            return []
        encoded = [data[0]]
        for i in range(1, len(data)):
            encoded.append(data[i] - data[i-1])
        return encoded

    @staticmethod
    def delta_decode(encoded: List[float]) -> List[float]:
        """Delta解码"""
        if not encoded:
            return []
        decoded = [encoded[0]]
        for i in range(1, len(encoded)):
            decoded.append(decoded[-1] + encoded[i])
        return decoded

    @staticmethod
    def run_length_encode(data: List[Any]) -> List[Tuple[Any, int]]:
        """Run-Length编码"""
        if not data:
            return []
        encoded = []
        current = data[0]
        count = 1
        for item in data[1:]:
            if item == current:
                count += 1
            else:
                encoded.append((current, count))
                current = item
                count = 1
        encoded.append((current, count))
        return encoded

    @staticmethod
    def run_length_decode(encoded: List[Tuple[Any, int]]) -> List[Any]:
        """Run-Length解码"""
        decoded = []
        for value, count in encoded:
            decoded.extend([value] * count)
        return decoded


# =============================================================================
# IoT设备 (IoTDevice)
# =============================================================================

class CommunicationProtocol(Enum):
    """通信协议"""
    MQTT = auto()
    COAP = auto()
    HTTP = auto()


class DeviceState(Enum):
    """设备状态"""
    SLEEP = auto()
    ACTIVE = auto()
    TRANSMITTING = auto()
    OFFLINE = auto()


class IoTDevice:
    """
    IoT设备模拟
    支持多种传感器、通信协议和电源管理
    """

    def __init__(self, device_id: str, config: IoTConfig):
        self.device_id = device_id
        self.config = config
        self.sensors: Dict[SensorType, Sensor] = {}
        self.protocol = CommunicationProtocol.MQTT
        self.state = DeviceState.SLEEP
        self.battery_level = 100.0  # percentage
        self.battery_capacity = config.default_battery_capacity  # mAh
        self.sampling_rate = config.default_sampling_rate
        self.last_sample_time = 0.0
        self.data_buffer: List[Dict] = []
        self.buffer_lock = threading.Lock()
        self.connected = False
        self.message_queue: deque = deque(maxlen=config.max_queue_size)
        self.statistics = {
            'samples_collected': 0,
            'messages_sent': 0,
            'bytes_transmitted': 0,
            'sleep_time': 0.0,
            'active_time': 0.0
        }
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_sensor(self, sensor_type: SensorType) -> Sensor:
        """添加传感器"""
        sensor = Sensor(sensor_type, self.config)
        self.sensors[sensor_type] = sensor
        return sensor

    def set_protocol(self, protocol: CommunicationProtocol):
        """设置通信协议"""
        self.protocol = protocol

    def connect(self) -> bool:
        """连接到网络"""
        if self.battery_level <= 0:
            return False
        self.connected = True
        self._transition_state(DeviceState.ACTIVE)
        return True

    def disconnect(self):
        """断开连接"""
        self.connected = False
        self._transition_state(DeviceState.OFFLINE)

    def start(self):
        """启动设备采样循环"""
        self._running = True
        self._thread = threading.Thread(target=self._sampling_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """停止设备"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _sampling_loop(self):
        """采样主循环"""
        while self._running:
            current_time = time.time()

            # 检查采样间隔
            if current_time - self.last_sample_time >= 1.0 / self.sampling_rate:
                self._sample_all_sensors(current_time)
                self.last_sample_time = current_time

            # 检查是否需要传输
            if len(self.data_buffer) >= 10 and self.connected:
                self._transmit_buffer()

            # 电源管理 - 进入休眠
            if self.state == DeviceState.ACTIVE:
                self._transition_state(DeviceState.SLEEP)

            time.sleep(0.01)

    def _sample_all_sensors(self, timestamp: float):
        """采样所有传感器"""
        self._transition_state(DeviceState.ACTIVE)

        readings = []
        for sensor_type, sensor in self.sensors.items():
            reading = sensor.read(timestamp)
            reading['device_id'] = self.device_id
            readings.append(reading)

        with self.buffer_lock:
            self.data_buffer.extend(readings)

        self.statistics['samples_collected'] += len(readings)
        self._consume_energy('active', 1.0 / self.sampling_rate)

    def _transmit_buffer(self):
        """传输缓冲区数据"""
        if not self.connected or not self.data_buffer:
            return

        self._transition_state(DeviceState.TRANSMITTING)

        with self.buffer_lock:
            data_to_send = self.data_buffer.copy()
            self.data_buffer.clear()

        # 模拟数据包封装
        packet = self._create_packet(data_to_send)
        packet_size = len(json.dumps(packet).encode('utf-8'))

        # 添加到消息队列
        self.message_queue.append({
            'timestamp': time.time(),
            'packet': packet,
            'size': packet_size
        })

        self.statistics['messages_sent'] += 1
        self.statistics['bytes_transmitted'] += packet_size

        # 传输能耗
        transmission_time = packet_size / 1000.0  # 假设1KB/s
        self._consume_energy('transmit', transmission_time)

    def _create_packet(self, data: List[Dict]) -> Dict:
        """创建数据包"""
        return {
            'device_id': self.device_id,
            'protocol': self.protocol.name,
            'timestamp': time.time(),
            'sequence': self.statistics['messages_sent'],
            'battery': self.battery_level,
            'data': data
        }

    def _transition_state(self, new_state: DeviceState):
        """状态转换"""
        if self.state == new_state:
            return

        current_time = time.time()
        # 记录上一状态的时间
        if self.state == DeviceState.SLEEP:
            self.statistics['sleep_time'] += current_time - getattr(self, '_state_start_time', current_time)
        elif self.state == DeviceState.ACTIVE:
            self.statistics['active_time'] += current_time - getattr(self, '_state_start_time', current_time)

        self.state = new_state
        self._state_start_time = current_time

    def _consume_energy(self, mode: str, duration: float):
        """消耗电池能量"""
        if mode == 'sleep':
            power = self.config.sleep_power
        elif mode == 'active':
            power = self.config.active_power
        elif mode == 'transmit':
            power = self.config.transmission_power
        else:
            power = self.config.active_power

        # 计算消耗的容量 (mAh)
        energy_consumed = (power * duration) / 1000.0  # 转换为 mWh
        capacity_consumed = energy_consumed / 3.7  # 假设3.7V电池

        # 更新电池百分比
        self.battery_level -= (capacity_consumed / self.battery_capacity) * 100
        self.battery_level = max(0.0, self.battery_level)

        if self.battery_level <= 0:
            self.disconnect()

    def get_status(self) -> Dict[str, Any]:
        """获取设备状态"""
        return {
            'device_id': self.device_id,
            'state': self.state.name,
            'battery_level': round(self.battery_level, 2),
            'connected': self.connected,
            'sensors': [s.name for s in self.sensors.keys()],
            'buffer_size': len(self.data_buffer),
            'queue_size': len(self.message_queue),
            'statistics': self.statistics.copy()
        }


# =============================================================================
# IoT网关 (IoTGateway)
# =============================================================================

class IoTGateway:
    """
    IoT网关/桥接
    支持协议转换、数据聚合、本地预处理和存储转发
    """

    def __init__(self, gateway_id: str, config: IoTConfig):
        self.gateway_id = gateway_id
        self.config = config
        self.devices: Dict[str, IoTDevice] = {}
        self.protocol_handlers: Dict[CommunicationProtocol, Callable] = {}
        self.aggregation_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self.aggregation_timer = 0.0
        self.store_forward_queue: deque = deque(maxlen=config.max_queue_size)
        self.filters: List[Callable[[Dict], bool]] = []
        self.preprocessors: List[Callable[[Dict], Dict]] = []
        self.statistics = {
            'packets_received': 0,
            'packets_forwarded': 0,
            'packets_filtered': 0,
            'bytes_aggregated': 0
        }
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """设置默认协议处理器"""
        self.protocol_handlers[CommunicationProtocol.MQTT] = self._handle_mqtt
        self.protocol_handlers[CommunicationProtocol.COAP] = self._handle_coap
        self.protocol_handlers[CommunicationProtocol.HTTP] = self._handle_http

    def register_device(self, device: IoTDevice):
        """注册设备到网关"""
        self.devices[device.device_id] = device

    def unregister_device(self, device_id: str):
        """注销设备"""
        if device_id in self.devices:
            del self.devices[device_id]

    def add_filter(self, filter_func: Callable[[Dict], bool]):
        """添加数据过滤器"""
        self.filters.append(filter_func)

    def add_preprocessor(self, preprocessor: Callable[[Dict], Dict]):
        """添加预处理器"""
        self.preprocessors.append(preprocessor)

    def receive_packet(self, packet: Dict) -> bool:
        """接收数据包"""
        self.statistics['packets_received'] += 1

        # 协议处理
        protocol = CommunicationProtocol[packet.get('protocol', 'MQTT')]
        handler = self.protocol_handlers.get(protocol)
        if handler:
            packet = handler(packet)

        # 应用过滤器
        for filter_func in self.filters:
            if not filter_func(packet):
                self.statistics['packets_filtered'] += 1
                return False

        # 应用预处理器
        for preprocessor in self.preprocessors:
            packet = preprocessor(packet)

        # 添加到聚合缓冲区
        device_id = packet.get('device_id', 'unknown')
        self.aggregation_buffer[device_id].append(packet)

        return True

    def _handle_mqtt(self, packet: Dict) -> Dict:
        """处理MQTT协议数据"""
        # MQTT特定处理 (QoS, 保留消息等)
        packet['qos'] = packet.get('qos', 1)
        return packet

    def _handle_coap(self, packet: Dict) -> Dict:
        """处理CoAP协议数据"""
        # CoAP特定处理 (确认消息等)
        packet['confirmable'] = packet.get('confirmable', True)
        return packet

    def _handle_http(self, packet: Dict) -> Dict:
        """处理HTTP协议数据"""
        packet['method'] = packet.get('method', 'POST')
        return packet

    def aggregate_and_forward(self) -> List[Dict]:
        """
        聚合并转发数据

        Returns:
            聚合后的数据包列表
        """
        aggregated_packets = []

        for device_id, packets in self.aggregation_buffer.items():
            if not packets:
                continue

            # 执行聚合
            aggregated = self._aggregate_packets(device_id, packets)
            aggregated_packets.append(aggregated)

            self.statistics['bytes_aggregated'] += len(json.dumps(aggregated).encode('utf-8'))
            self.statistics['packets_forwarded'] += 1

        # 清空缓冲区
        self.aggregation_buffer.clear()

        return aggregated_packets

    def _aggregate_packets(self, device_id: str, packets: List[Dict]) -> Dict:
        """聚合数据包"""
        if not packets:
            return {}

        # 提取所有传感器数据
        all_data = []
        for packet in packets:
            all_data.extend(packet.get('data', []))

        # 按传感器类型分组统计
        sensor_stats = defaultdict(lambda: {'values': [], 'timestamps': []})
        for reading in all_data:
            sensor_type = reading.get('sensor_type', 'unknown')
            sensor_stats[sensor_type]['values'].append(reading.get('value', 0))
            sensor_stats[sensor_type]['timestamps'].append(reading.get('timestamp', 0))

        # 计算统计量
        aggregated_data = {}
        for sensor_type, stats in sensor_stats.items():
            values = stats['values']
            if values:
                aggregated_data[sensor_type] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': self._calculate_std(values),
                    'start_time': min(stats['timestamps']),
                    'end_time': max(stats['timestamps'])
                }

        return {
            'gateway_id': self.gateway_id,
            'device_id': device_id,
            'timestamp': time.time(),
            'packet_count': len(packets),
            'aggregated_data': aggregated_data
        }

    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def store_and_forward(self, packet: Dict) -> bool:
        """存储转发模式"""
        try:
            self.store_forward_queue.append({
                'timestamp': time.time(),
                'packet': packet,
                'retry_count': 0
            })
            return True
        except:
            return False

    def process_store_forward(self, forward_func: Callable[[Dict], bool]) -> int:
        """处理存储转发队列"""
        processed = 0
        temp_queue = deque()

        while self.store_forward_queue:
            item = self.store_forward_queue.popleft()

            if forward_func(item['packet']):
                processed += 1
            else:
                # 重试计数
                item['retry_count'] += 1
                if item['retry_count'] < 3:
                    temp_queue.append(item)

        self.store_forward_queue = temp_queue
        return processed


# =============================================================================
# 数据流处理 (DataStreaming)
# =============================================================================

class DataStreaming:
    """
    流数据处理
    支持滑动窗口、事件检测、流聚合和复杂事件处理(CEP)
    """

    def __init__(self, config: IoTConfig):
        self.config = config
        self.windows: Dict[str, deque] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.cep_patterns: Dict[str, Any] = {}
        self.cep_states: Dict[str, Any] = {}

    def create_window(self, window_id: str, size: int):
        """创建滑动窗口"""
        self.windows[window_id] = deque(maxlen=size)

    def add_to_window(self, window_id: str, data: Dict):
        """添加数据到窗口"""
        if window_id not in self.windows:
            self.create_window(window_id, self.config.window_size)
        self.windows[window_id].append(data)

    def sliding_window_operation(self, window_id: str, operation: str) -> Optional[float]:
        """
        滑动窗口操作

        Args:
            window_id: 窗口ID
            operation: 操作类型 (avg, min, max, sum, count, std)

        Returns:
            操作结果
        """
        if window_id not in self.windows or not self.windows[window_id]:
            return None

        window = list(self.windows[window_id])
        values = [d.get('value', 0) for d in window if 'value' in d]

        if not values:
            return None

        if operation == 'avg':
            return sum(values) / len(values)
        elif operation == 'min':
            return min(values)
        elif operation == 'max':
            return max(values)
        elif operation == 'sum':
            return sum(values)
        elif operation == 'count':
            return len(values)
        elif operation == 'std':
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return math.sqrt(variance)

        return None

    def detect_threshold_event(self, data: Dict, threshold: float,
                               comparison: str = 'above') -> bool:
        """
        阈值事件检测

        Args:
            data: 数据点
            threshold: 阈值
            comparison: 比较类型 ('above', 'below', 'equal')

        Returns:
            是否触发事件
        """
        value = data.get('value', 0)

        if comparison == 'above':
            return value > threshold
        elif comparison == 'below':
            return value < threshold
        elif comparison == 'equal':
            return abs(value - threshold) < 0.001

        return False

    def detect_anomaly_event(self, window_id: str, data: Dict,
                             method: str = 'zscore') -> Tuple[bool, float]:
        """
        异常事件检测

        Args:
            window_id: 窗口ID
            data: 当前数据点
            method: 检测方法 ('zscore', 'iqr', 'mad')

        Returns:
            (是否异常, 异常分数)
        """
        if window_id not in self.windows or len(self.windows[window_id]) < 10:
            return False, 0.0

        window = list(self.windows[window_id])
        values = [d.get('value', 0) for d in window]
        current_value = data.get('value', 0)

        if method == 'zscore':
            mean = sum(values) / len(values)
            std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
            if std == 0:
                return False, 0.0
            zscore = abs(current_value - mean) / std
            return zscore > self.config.anomaly_threshold, zscore

        elif method == 'iqr':
            sorted_values = sorted(values)
            q1_idx = len(sorted_values) // 4
            q3_idx = 3 * len(sorted_values) // 4
            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            is_anomaly = current_value < lower or current_value > upper
            score = max(abs(current_value - lower), abs(current_value - upper)) / iqr if iqr > 0 else 0
            return is_anomaly, score

        elif method == 'mad':
            median = sorted(values)[len(values) // 2]
            mad_values = [abs(x - median) for x in values]
            mad = sorted(mad_values)[len(mad_values) // 2]
            if mad == 0:
                return False, 0.0
            score = abs(current_value - median) / mad
            return score > 3.5, score

        return False, 0.0

    def register_event_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self.event_handlers[event_type].append(handler)

    def emit_event(self, event_type: str, event_data: Dict):
        """触发事件"""
        for handler in self.event_handlers[event_type]:
            handler(event_data)

    def define_cep_pattern(self, pattern_id: str, pattern: Dict):
        """
        定义CEP模式

        pattern格式:
        {
            'sequence': [
                {'event_type': 'temp_high', 'condition': lambda e: e['value'] > 30},
                {'event_type': 'temp_low', 'condition': lambda e: e['value'] < 20}
            ],
            'time_window': 60  # seconds
        }
        """
        self.cep_patterns[pattern_id] = pattern
        self.cep_states[pattern_id] = {
            'matched_events': [],
            'start_time': None
        }

    def process_cep(self, event: Dict) -> List[str]:
        """
        处理复杂事件模式

        Args:
            event: 输入事件

        Returns:
            匹配的模式ID列表
        """
        matched_patterns = []
        current_time = event.get('timestamp', time.time())

        for pattern_id, pattern in self.cep_patterns.items():
            state = self.cep_states[pattern_id]
            sequence = pattern['sequence']
            time_window = pattern.get('time_window', 60)

            # 检查当前事件是否匹配序列中的下一个模式
            next_step = len(state['matched_events'])
            if next_step >= len(sequence):
                continue

            expected = sequence[next_step]
            if event.get('event_type') == expected['event_type']:
                if expected.get('condition', lambda e: True)(event):
                    if state['start_time'] is None:
                        state['start_time'] = current_time

                    # 检查时间窗口
                    if current_time - state['start_time'] <= time_window:
                        state['matched_events'].append(event)

                        # 检查是否完成整个序列
                        if len(state['matched_events']) == len(sequence):
                            matched_patterns.append(pattern_id)
                            # 重置状态
                            state['matched_events'] = []
                            state['start_time'] = None
                    else:
                        # 超时,重置
                        state['matched_events'] = [event]
                        state['start_time'] = current_time

        return matched_patterns


# =============================================================================
# 时序数据库 (TimeSeriesDatabase)
# =============================================================================

class TimeSeriesDatabase:
    """
    时序数据库
    支持降采样、保留策略、时间索引和查询优化
    """

    def __init__(self, config: IoTConfig):
        self.config = config
        self.data: Dict[str, List[Dict]] = defaultdict(list)
        self.downsampled_data: Dict[str, Dict[int, List[Dict]]] = defaultdict(dict)
        self.index: Dict[str, Dict] = defaultdict(lambda: {'min_time': float('inf'), 'max_time': 0})
        self.retention_policies: Dict[str, int] = {}  # series_id -> retention_days
        self.query_cache: Dict[str, Tuple[List[Dict], float]] = {}
        self.cache_ttl = 60.0  # seconds

    def write(self, series_id: str, timestamp: float, value: float, tags: Dict = None):
        """
        写入时序数据

        Args:
            series_id: 序列ID
            timestamp: 时间戳
            value: 值
            tags: 标签
        """
        point = {
            'timestamp': timestamp,
            'value': value,
            'tags': tags or {}
        }

        self.data[series_id].append(point)

        # 更新索引
        self.index[series_id]['min_time'] = min(self.index[series_id]['min_time'], timestamp)
        self.index[series_id]['max_time'] = max(self.index[series_id]['max_time'], timestamp)

        # 检查降采样
        self._check_downsampling(series_id)

    def query(self, series_id: str, start_time: float, end_time: float,
              aggregation: str = None, interval: int = None) -> List[Dict]:
        """
        查询时序数据

        Args:
            series_id: 序列ID
            start_time: 开始时间
            end_time: 结束时间
            aggregation: 聚合方式 (avg, min, max, sum, count)
            interval: 聚合间隔 (秒)

        Returns:
            查询结果
        """
        cache_key = f"{series_id}:{start_time}:{end_time}:{aggregation}:{interval}"

        # 检查缓存
        if cache_key in self.query_cache:
            result, cache_time = self.query_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return result

        # 选择数据源 (原始或降采样)
        data_source = self._select_data_source(series_id, interval)

        # 过滤时间范围
        filtered = [
            p for p in data_source
            if start_time <= p['timestamp'] <= end_time
        ]

        # 应用聚合
        if aggregation and interval:
            filtered = self._aggregate_by_interval(filtered, aggregation, interval)

        # 更新缓存
        self.query_cache[cache_key] = (filtered, time.time())

        return filtered

    def _select_data_source(self, series_id: str, interval: Optional[int]) -> List[Dict]:
        """选择合适的数据源"""
        if interval is None:
            return self.data[series_id]

        # 查找最接近的降采样数据
        best_interval = None
        for ds_interval in self.downsampled_data.get(series_id, {}).keys():
            if ds_interval <= interval:
                if best_interval is None or ds_interval > best_interval:
                    best_interval = ds_interval

        if best_interval:
            return self.downsampled_data[series_id][best_interval]

        return self.data[series_id]

    def _aggregate_by_interval(self, data: List[Dict], aggregation: str,
                               interval: int) -> List[Dict]:
        """按间隔聚合数据"""
        if not data:
            return []

        # 按间隔分组
        buckets = defaultdict(list)
        for point in data:
            bucket_time = (int(point['timestamp']) // interval) * interval
            buckets[bucket_time].append(point['value'])

        # 聚合每个桶
        result = []
        for bucket_time in sorted(buckets.keys()):
            values = buckets[bucket_time]
            if aggregation == 'avg':
                agg_value = sum(values) / len(values)
            elif aggregation == 'min':
                agg_value = min(values)
            elif aggregation == 'max':
                agg_value = max(values)
            elif aggregation == 'sum':
                agg_value = sum(values)
            elif aggregation == 'count':
                agg_value = len(values)
            else:
                agg_value = sum(values) / len(values)

            result.append({
                'timestamp': bucket_time,
                'value': agg_value,
                'count': len(values)
            })

        return result

    def _check_downsampling(self, series_id: str):
        """检查并执行降采样"""
        data = self.data[series_id]
        if len(data) < 1000:
            return

        for interval in self.config.downsampling_intervals:
            if series_id not in self.downsampled_data:
                self.downsampled_data[series_id] = {}

            if interval not in self.downsampled_data[series_id]:
                self.downsampled_data[series_id][interval] = []

            # 降采样
            last_bucket = None
            bucket_values = []

            for point in data:
                bucket_time = (int(point['timestamp']) // interval) * interval

                if bucket_time != last_bucket:
                    if bucket_values:
                        # 保存上一个桶的平均值
                        self.downsampled_data[series_id][interval].append({
                            'timestamp': last_bucket,
                            'value': sum(bucket_values) / len(bucket_values),
                            'count': len(bucket_values)
                        })
                    bucket_values = []
                    last_bucket = bucket_time

                bucket_values.append(point['value'])

    def apply_retention_policy(self, series_id: str, retention_days: int):
        """应用保留策略"""
        self.retention_policies[series_id] = retention_days
        cutoff_time = time.time() - (retention_days * 24 * 3600)

        # 清理过期数据
        self.data[series_id] = [
            p for p in self.data[series_id]
            if p['timestamp'] >= cutoff_time
        ]

        # 清理降采样数据
        if series_id in self.downsampled_data:
            for interval in self.downsampled_data[series_id]:
                self.downsampled_data[series_id][interval] = [
                    p for p in self.downsampled_data[series_id][interval]
                    if p['timestamp'] >= cutoff_time
                ]

    def get_statistics(self, series_id: str) -> Dict:
        """获取序列统计信息"""
        data = self.data[series_id]
        if not data:
            return {'count': 0}

        values = [p['value'] for p in data]
        return {
            'count': len(data),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'time_range': {
                'start': self.index[series_id]['min_time'],
                'end': self.index[series_id]['max_time']
            },
            'retention_days': self.retention_policies.get(series_id, self.config.retention_days)
        }


# =============================================================================
# 异常检测 (AnomalyDetection)
# =============================================================================

class AnomalyDetection:
    """
    IoT异常检测
    支持统计方法、机器学习方法、LSTM预测和集成方法
    """

    def __init__(self, config: IoTConfig):
        self.config = config
        self.models = {}
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.window_size * 10))
        self.lstm_states: Dict[str, Any] = {}

    # ============= 统计方法 =============

    def zscore_detection(self, data: List[float], threshold: float = None) -> List[Tuple[int, float]]:
        """
        Z-Score异常检测

        Returns:
            异常点列表 [(索引, Z-Score), ...]
        """
        if len(data) < 2:
            return []

        threshold = threshold or self.config.anomaly_threshold
        mean = sum(data) / len(data)
        std = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))

        if std == 0:
            return []

        anomalies = []
        for i, value in enumerate(data):
            zscore = abs(value - mean) / std
            if zscore > threshold:
                anomalies.append((i, zscore))

        return anomalies

    def iqr_detection(self, data: List[float]) -> List[int]:
        """
        IQR (四分位距) 异常检测

        Returns:
            异常点索引列表
        """
        if len(data) < 4:
            return []

        sorted_data = sorted(data)
        q1_idx = len(sorted_data) // 4
        q3_idx = 3 * len(sorted_data) // 4
        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        anomalies = []
        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                anomalies.append(i)

        return anomalies

    def moving_average_deviation(self, data: List[float], window: int = 10,
                                  threshold: float = 2.0) -> List[Tuple[int, float]]:
        """
        移动平均偏差检测

        Returns:
            异常点列表 [(索引, 偏差), ...]
        """
        if len(data) < window:
            return []

        anomalies = []
        for i in range(window, len(data)):
            window_data = data[i-window:i]
            ma = sum(window_data) / len(window_data)
            std = math.sqrt(sum((x - ma) ** 2 for x in window_data) / len(window_data))

            if std > 0:
                deviation = abs(data[i] - ma) / std
                if deviation > threshold:
                    anomalies.append((i, deviation))

        return anomalies

    # ============= 机器学习方法 =============

    def isolation_forest_simulation(self, data: List[List[float]],
                                     n_trees: int = 10,
                                     sample_size: int = 256) -> List[float]:
        """
        隔离森林异常检测 (简化模拟)

        Args:
            data: 数据点列表 [[特征1, 特征2, ...], ...]
            n_trees: 树的数量
            sample_size: 采样大小

        Returns:
            异常分数列表 (越高越异常)
        """
        if not data:
            return []

        n_samples = len(data)
        scores = []

        for point in data:
            # 简化的隔离分数计算
            # 基于到最近邻居的距离
            distances = []
            for other in data:
                if other != point:
                    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(point, other)))
                    distances.append(dist)

            if distances:
                min_dist = min(distances)
                avg_dist = sum(distances) / len(distances)
                # 距离越远,异常分数越高
                score = min_dist / (avg_dist + 1e-10)
                scores.append(min(1.0, score))
            else:
                scores.append(0.0)

        return scores

    def one_class_svm_simulation(self, data: List[List[float]],
                                  nu: float = 0.1) -> List[int]:
        """
        One-Class SVM异常检测 (简化模拟)

        Returns:
            预测标签列表 (1=正常, -1=异常)
        """
        if not data:
            return []

        # 计算数据中心的近似
        n_features = len(data[0])
        center = [sum(d[i] for d in data) / len(data) for i in range(n_features)]

        # 计算平均距离作为阈值
        distances = []
        for point in data:
            dist = math.sqrt(sum((point[i] - center[i]) ** 2 for i in range(n_features)))
            distances.append(dist)

        threshold = sorted(distances)[int(len(distances) * (1 - nu))]

        # 预测
        predictions = []
        for point in data:
            dist = math.sqrt(sum((point[i] - center[i]) ** 2 for i in range(n_features)))
            predictions.append(1 if dist <= threshold else -1)

        return predictions

    # ============= LSTM预测 =============

    def lstm_predict(self, series_id: str, sequence: List[float],
                     prediction_steps: int = 1) -> List[float]:
        """
        LSTM预测 (简化模拟)

        Args:
            series_id: 序列ID
            sequence: 输入序列
            prediction_steps: 预测步数

        Returns:
            预测值列表
        """
        # 简化的LSTM模拟 - 使用指数加权移动平均
        if len(sequence) < 2:
            return [sequence[-1]] * prediction_steps if sequence else [0.0]

        # 更新历史数据
        self.historical_data[series_id].extend(sequence)

        # 使用指数平滑进行预测
        alpha = 0.3
        predictions = []
        last_value = sequence[-1]

        for _ in range(prediction_steps):
            # 计算趋势
            if len(sequence) >= 2:
                trend = sequence[-1] - sequence[-2]
            else:
                trend = 0

            # 指数平滑 + 趋势
            prediction = alpha * last_value + (1 - alpha) * (last_value + trend)
            predictions.append(prediction)
            last_value = prediction

        return predictions

    def detect_with_lstm(self, series_id: str, current_value: float,
                         threshold: float = 2.0) -> Tuple[bool, float]:
        """
        使用LSTM预测进行异常检测

        Returns:
            (是否异常, 异常分数)
        """
        history = list(self.historical_data[series_id])
        if len(history) < 10:
            return False, 0.0

        # 预测下一个值
        prediction = self.lstm_predict(series_id, history[-10:], 1)[0]

        # 计算预测误差
        error = abs(current_value - prediction)

        # 计算历史预测误差的标准差
        recent_errors = []
        for i in range(10, min(50, len(history))):
            pred = self.lstm_predict(series_id, history[i-10:i], 1)[0]
            err = abs(history[i] - pred)
            recent_errors.append(err)

        if not recent_errors:
            return False, 0.0

        mean_error = sum(recent_errors) / len(recent_errors)
        std_error = math.sqrt(sum((e - mean_error) ** 2 for e in recent_errors) / len(recent_errors))

        if std_error == 0:
            return False, 0.0

        anomaly_score = (error - mean_error) / std_error
        is_anomaly = anomaly_score > threshold

        return is_anomaly, anomaly_score

    # ============= 集成方法 =============

    def ensemble_detection(self, data: List[float]) -> Dict[str, Any]:
        """
        集成异常检测

        组合多种检测方法,通过投票决定最终结果

        Returns:
            检测结果字典
        """
        if len(data) < 10:
            return {'anomaly': False, 'confidence': 0.0, 'methods': {}}

        results = {}
        votes = []

        # Z-Score检测
        zscore_anomalies = self.zscore_detection(data)
        zscore_detected = len(zscore_anomalies) > 0
        results['zscore'] = {'detected': zscore_detected, 'count': len(zscore_anomalies)}
        votes.append(1 if zscore_detected else 0)

        # IQR检测
        iqr_anomalies = self.iqr_detection(data)
        iqr_detected = len(iqr_anomalies) > 0
        results['iqr'] = {'detected': iqr_detected, 'count': len(iqr_anomalies)}
        votes.append(1 if iqr_detected else 0)

        # 移动平均偏差检测
        ma_anomalies = self.moving_average_deviation(data)
        ma_detected = len(ma_anomalies) > 0
        results['moving_average'] = {'detected': ma_detected, 'count': len(ma_anomalies)}
        votes.append(1 if ma_detected else 0)

        # 投票结果
        vote_sum = sum(votes)
        confidence = vote_sum / len(votes)
        is_anomaly = vote_sum >= len(votes) / 2

        return {
            'anomaly': is_anomaly,
            'confidence': confidence,
            'methods': results,
            'vote_ratio': vote_sum / len(votes)
        }


# =============================================================================
# 数字孪生 (DigitalTwin)
# =============================================================================

class DigitalTwin:
    """
    数字孪生模拟
    支持物理建模、状态估计、预测和What-if分析
    """

    def __init__(self, twin_id: str, config: IoTConfig):
        self.twin_id = twin_id
        self.config = config
        self.physical_model = {}
        self.state = {}
        self.kalman_state = None
        self.kalman_covariance = None
        self.history: deque = deque(maxlen=1000)
        self.predictions: List[Dict] = []

    def define_physical_model(self, model_params: Dict):
        """
        定义物理模型

        model_params示例:
        {
            'type': 'thermal',
            'thermal_mass': 1000,  # J/K
            'heat_transfer_coeff': 10,  # W/K
            'ambient_temp': 25  # °C
        }
        """
        self.physical_model = model_params

    def update_state(self, measurements: Dict[str, float]):
        """
        更新数字孪生状态

        Args:
            measurements: 传感器测量值
        """
        timestamp = time.time()

        # 使用卡尔曼滤波更新状态
        estimated_state = self._kalman_update(measurements)

        state_record = {
            'timestamp': timestamp,
            'measurements': measurements.copy(),
            'estimated_state': estimated_state,
            'model': self.physical_model.copy()
        }

        self.state = estimated_state
        self.history.append(state_record)

    def _kalman_update(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        卡尔曼滤波状态估计

        简化的卡尔曼滤波实现
        """
        if self.kalman_state is None:
            # 初始化
            self.kalman_state = {k: v for k, v in measurements.items()}
            self.kalman_covariance = {k: 1.0 for k in measurements.keys()}
            return self.kalman_state.copy()

        estimated = {}
        Q = self.config.kalman_process_noise  # 过程噪声
        R = self.config.kalman_measurement_noise  # 测量噪声

        for key, measurement in measurements.items():
            # 预测
            x_pred = self.kalman_state.get(key, measurement)
            P_pred = self.kalman_covariance.get(key, 1.0) + Q

            # 更新
            K = P_pred / (P_pred + R)  # 卡尔曼增益
            x_est = x_pred + K * (measurement - x_pred)
            P_est = (1 - K) * P_pred

            estimated[key] = x_est
            self.kalman_state[key] = x_est
            self.kalman_covariance[key] = P_est

        return estimated

    def simulate(self, duration: float, time_step: float = 1.0) -> List[Dict]:
        """
        运行物理模型仿真

        Args:
            duration: 仿真时长 (秒)
            time_step: 时间步长 (秒)

        Returns:
            仿真结果列表
        """
        if not self.physical_model:
            return []

        results = []
        current_state = self.state.copy()
        steps = int(duration / time_step)

        for step in range(steps):
            # 根据模型类型进行仿真
            model_type = self.physical_model.get('type', 'generic')

            if model_type == 'thermal':
                current_state = self._simulate_thermal(current_state, time_step)
            elif model_type == 'mechanical':
                current_state = self._simulate_mechanical(current_state, time_step)
            else:
                current_state = self._simulate_generic(current_state, time_step)

            results.append({
                'step': step,
                'time': step * time_step,
                'state': current_state.copy()
            })

        self.predictions = results
        return results

    def _simulate_thermal(self, state: Dict, dt: float) -> Dict:
        """热力学模型仿真"""
        thermal_mass = self.physical_model.get('thermal_mass', 1000)
        heat_transfer = self.physical_model.get('heat_transfer_coeff', 10)
        ambient = self.physical_model.get('ambient_temp', 25)

        current_temp = state.get('temperature', ambient)

        # 牛顿冷却定律
        dT = -heat_transfer * (current_temp - ambient) / thermal_mass * dt
        new_temp = current_temp + dT

        return {**state, 'temperature': new_temp}

    def _simulate_mechanical(self, state: Dict, dt: float) -> Dict:
        """机械系统仿真"""
        mass = self.physical_model.get('mass', 1.0)
        damping = self.physical_model.get('damping', 0.1)
        spring_k = self.physical_model.get('spring_constant', 10)

        position = state.get('position', 0)
        velocity = state.get('velocity', 0)

        # 简谐振动
        acceleration = -(spring_k * position + damping * velocity) / mass
        new_velocity = velocity + acceleration * dt
        new_position = position + velocity * dt

        return {
            **state,
            'position': new_position,
            'velocity': new_velocity,
            'acceleration': acceleration
        }

    def _simulate_generic(self, state: Dict, dt: float) -> Dict:
        """通用模型仿真"""
        # 简单的随机游走模型
        new_state = {}
        for key, value in state.items():
            noise = random.gauss(0, 0.1)
            new_state[key] = value + noise * dt
        return new_state

    def what_if_analysis(self, scenario: Dict, duration: float) -> List[Dict]:
        """
        What-if分析

        Args:
            scenario: 场景参数
                {
                    'parameter_changes': {'temperature_setpoint': 30},
                    'disturbances': [{'time': 10, 'type': 'step', 'magnitude': 5}]
                }
            duration: 仿真时长

        Returns:
            分析结果
        """
        # 保存当前状态
        original_model = self.physical_model.copy()
        original_state = self.state.copy()

        # 应用场景参数
        if 'parameter_changes' in scenario:
            self.physical_model.update(scenario['parameter_changes'])

        # 运行仿真
        results = self.simulate(duration)

        # 应用干扰
        if 'disturbances' in scenario:
            for disturbance in scenario['disturbances']:
                step_idx = int(disturbance['time'] / 1.0)
                if 0 <= step_idx < len(results):
                    if disturbance['type'] == 'step':
                        for key in results[step_idx]['state']:
                            results[step_idx]['state'][key] += disturbance['magnitude']

        # 恢复原始状态
        self.physical_model = original_model
        self.state = original_state

        return results

    def compare_with_physical(self, physical_measurements: List[Dict]) -> Dict:
        """
        比较数字孪生与物理实体的差异

        Returns:
            差异分析结果
        """
        if not self.history or not physical_measurements:
            return {'error': 'Insufficient data'}

        differences = []
        for i, physical in enumerate(physical_measurements):
            if i < len(self.history):
                twin_state = self.history[i]['estimated_state']
                diff = {}
                for key in physical:
                    if key in twin_state:
                        diff[key] = abs(physical[key] - twin_state[key])
                differences.append(diff)

        # 计算统计差异
        if not differences:
            return {'error': 'No comparable data'}

        stats = {}
        all_keys = set()
        for diff in differences:
            all_keys.update(diff.keys())

        for key in all_keys:
            values = [d.get(key, 0) for d in differences if key in d]
            if values:
                stats[key] = {
                    'mean_diff': sum(values) / len(values),
                    'max_diff': max(values),
                    'std_diff': math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / len(values))
                }

        return {
            'differences': differences,
            'statistics': stats,
            'sample_count': len(differences)
        }


# =============================================================================
# IoT安全 (IoTSecurity)
# =============================================================================

class IoTSecurity:
    """
    IoT安全机制
    支持轻量级加密、认证、访问控制和入侵检测
    """

    def __init__(self, config: IoTConfig):
        self.config = config
        self.tokens: Dict[str, Dict] = {}  # token -> {user_id, expiry, permissions}
        self.acl: Dict[str, List[str]] = defaultdict(list)  # resource -> [allowed_users]
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)  # user_id -> [timestamps]
        self.intrusion_patterns: List[Dict] = []
        self.security_log: deque = deque(maxlen=1000)

    # ============= 轻量级加密 =============

    def xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """XOR加密 (轻量级)"""
        return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

    def xor_decrypt(self, data: bytes, key: bytes) -> bytes:
        """XOR解密"""
        return self.xor_encrypt(data, key)  # XOR是对称的

    def simple_aes_like_encrypt(self, data: bytes, key: bytes, rounds: int = 4) -> bytes:
        """
        简化AES-like加密

        这是一个教育性实现,展示分组密码的基本概念
        """
        block_size = 16
        key_size = 16

        # 填充
        padding = block_size - (len(data) % block_size)
        padded = data + bytes([padding] * padding)

        # 密钥调度 (简化)
        round_keys = self._key_schedule(key[:key_size], rounds)

        # 分组加密
        encrypted = b''
        for i in range(0, len(padded), block_size):
            block = padded[i:i+block_size]
            encrypted_block = self._encrypt_block(block, round_keys, rounds)
            encrypted += encrypted_block

        return encrypted

    def simple_aes_like_decrypt(self, data: bytes, key: bytes, rounds: int = 4) -> bytes:
        """简化AES-like解密"""
        block_size = 16
        key_size = 16

        round_keys = self._key_schedule(key[:key_size], rounds)

        decrypted = b''
        for i in range(0, len(data), block_size):
            block = data[i:i+block_size]
            decrypted_block = self._decrypt_block(block, round_keys, rounds)
            decrypted += decrypted_block

        # 去除填充
        padding = decrypted[-1]
        return decrypted[:-padding]

    def _key_schedule(self, key: bytes, rounds: int) -> List[bytes]:
        """密钥调度 (简化)"""
        round_keys = [key]
        for i in range(rounds):
            # 简单的密钥派生
            new_key = hashlib.sha256(round_keys[-1] + bytes([i])).digest()[:16]
            round_keys.append(new_key)
        return round_keys

    def _encrypt_block(self, block: bytes, round_keys: List[bytes], rounds: int) -> bytes:
        """加密单个块"""
        state = list(block)

        for r in range(rounds):
            # 轮密钥加
            state = [s ^ k for s, k in zip(state, round_keys[r])]
            # 替换层 (简化S-box)
            state = [(s * 3 + r) % 256 for s in state]
            # 置换层 (循环移位)
            state = state[4:] + state[:4]

        # 最终轮密钥加
        state = [s ^ k for s, k in zip(state, round_keys[rounds])]

        return bytes(state)

    def _decrypt_block(self, block: bytes, round_keys: List[bytes], rounds: int) -> bytes:
        """解密单个块"""
        state = list(block)

        # 逆向最终轮密钥加
        state = [s ^ k for s, k in zip(state, round_keys[rounds])]

        for r in range(rounds - 1, -1, -1):
            # 逆向置换
            state = state[-4:] + state[:-4]
            # 逆向替换
            state = [((s - r) * 171) % 256 for s in state]  # 3 * 171 = 513 = 1 (mod 256)
            # 逆向轮密钥加
            state = [s ^ k for s, k in zip(state, round_keys[r])]

        return bytes(state)

    # ============= 认证 =============

    def generate_token(self, user_id: str, permissions: List[str]) -> str:
        """生成认证令牌"""
        expiry = time.time() + self.config.token_expiry
        token = hashlib.sha256(f"{user_id}:{expiry}:{random.random()}".encode()).hexdigest()

        self.tokens[token] = {
            'user_id': user_id,
            'expiry': expiry,
            'permissions': permissions
        }

        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """验证令牌"""
        if token not in self.tokens:
            return None

        token_data = self.tokens[token]
        if time.time() > token_data['expiry']:
            del self.tokens[token]
            return None

        return token_data

    def revoke_token(self, token: str):
        """撤销令牌"""
        if token in self.tokens:
            del self.tokens[token]

    # ============= 访问控制 =============

    def grant_access(self, resource: str, user_id: str):
        """授予访问权限"""
        if user_id not in self.acl[resource]:
            self.acl[resource].append(user_id)

    def revoke_access(self, resource: str, user_id: str):
        """撤销访问权限"""
        if user_id in self.acl[resource]:
            self.acl[resource].remove(user_id)

    def check_access(self, resource: str, user_id: str) -> bool:
        """检查访问权限"""
        return user_id in self.acl.get(resource, [])

    # ============= 入侵检测 =============

    def log_access_attempt(self, user_id: str, resource: str, success: bool):
        """记录访问尝试"""
        timestamp = time.time()
        self.security_log.append({
            'timestamp': timestamp,
            'user_id': user_id,
            'resource': resource,
            'success': success
        })

        if not success:
            self.failed_attempts[user_id].append(timestamp)
            # 清理旧记录
            self.failed_attempts[user_id] = [
                t for t in self.failed_attempts[user_id]
                if timestamp - t < 3600  # 1小时内
            ]

    def detect_intrusion(self, user_id: str) -> Dict:
        """
        入侵检测

        检测异常访问模式:
        - 短时间内多次失败
        - 异常时间访问
        - 权限提升尝试
        """
        alerts = []
        current_time = time.time()

        # 检查失败尝试
        recent_failures = [
            t for t in self.failed_attempts.get(user_id, [])
            if current_time - t < 300  # 5分钟内
        ]

        if len(recent_failures) >= self.config.max_failed_auth:
            alerts.append({
                'type': 'brute_force',
                'severity': 'high',
                'message': f'User {user_id} has {len(recent_failures)} failed attempts in 5 minutes'
            })

        # 检查异常时间 (假设正常时间是8-18点)
        current_hour = datetime.now().hour
        if current_hour < 8 or current_hour > 18:
            alerts.append({
                'type': 'unusual_time',
                'severity': 'medium',
                'message': f'Access attempt during off-hours by {user_id}'
            })

        # 检查异常频率
        recent_logs = [
            log for log in self.security_log
            if log['user_id'] == user_id and current_time - log['timestamp'] < 60
        ]

        if len(recent_logs) > 100:  # 1分钟内超过100次访问
            alerts.append({
                'type': 'rate_anomaly',
                'severity': 'high',
                'message': f'Abnormal access rate by {user_id}: {len(recent_logs)} requests/minute'
            })

        return {
            'user_id': user_id,
            'alerts': alerts,
            'risk_score': len(alerts) * 0.3 + len(recent_failures) * 0.1,
            'timestamp': current_time
        }


# =============================================================================
# 能源管理 (EnergyManagement)
# =============================================================================

class EnergyManagement:
    """
    能源管理
    支持占空比控制、自适应采样、能量收集模型和电池寿命预测
    """

    def __init__(self, config: IoTConfig):
        self.config = config
        self.duty_cycle = 0.1  # 10% 占空比
        self.sampling_rate = config.default_sampling_rate
        self.energy_harvesting_rate = 0.0  # mW
        self.battery_capacity = config.default_battery_capacity  # mAh
        self.current_battery = self.battery_capacity
        self.energy_history: deque = deque(maxlen=1000)
        self.consumption_model = {
            'sleep': config.sleep_power,
            'active': config.active_power,
            'transmit': config.transmission_power
        }

    def set_duty_cycle(self, duty_cycle: float):
        """
        设置占空比

        Args:
            duty_cycle: 0-1之间的值,表示活跃时间比例
        """
        self.duty_cycle = max(0.01, min(1.0, duty_cycle))

    def calculate_duty_cycle_for_lifetime(self, target_days: float,
                                           daily_transmissions: int = 24) -> float:
        """
        计算实现目标寿命所需的占空比

        Args:
            target_days: 目标运行天数
            daily_transmissions: 每日传输次数

        Returns:
            建议占空比
        """
        # 总可用能量 (mWh)
        total_energy = self.battery_capacity * 3.7

        # 目标运行时间 (小时)
        target_hours = target_days * 24

        # 每次传输能耗
        energy_per_transmission = self.consumption_model['transmit'] * 1.0  # 假设1秒传输

        # 每日传输能耗
        daily_tx_energy = daily_transmissions * energy_per_transmission

        # 每日可用能耗
        daily_available = total_energy / target_days

        # 计算可用于活跃状态的能耗
        daily_active_energy = daily_available - daily_tx_energy

        if daily_active_energy <= 0:
            return 0.01  # 最小占空比

        # 计算占空比
        # 假设活跃状态平均功耗
        avg_active_power = (self.consumption_model['active'] + self.consumption_model['sleep']) / 2
        duty = daily_active_energy / (avg_active_power * 24)

        return max(0.01, min(1.0, duty))

    def adaptive_sampling(self, data_variance: float, importance_threshold: float = 0.5) -> float:
        """
        自适应采样率调整

        根据数据变化动态调整采样率

        Args:
            data_variance: 数据方差
            importance_threshold: 重要性阈值

        Returns:
            建议采样率 (Hz)
        """
        base_rate = self.config.default_sampling_rate

        # 方差高时增加采样率
        if data_variance > importance_threshold:
            new_rate = base_rate * (1 + data_variance)
        else:
            # 方差低时降低采样率
            new_rate = base_rate * max(0.1, data_variance / importance_threshold)

        self.sampling_rate = min(10.0, max(0.1, new_rate))  # 限制在0.1-10Hz
        return self.sampling_rate

    def simulate_energy_harvesting(self, source_type: str, conditions: Dict) -> float:
        """
        模拟能量收集

        Args:
            source_type: 能量源类型 ('solar', 'thermal', 'vibration', 'rf')
            conditions: 环境条件

        Returns:
            收集功率 (mW)
        """
        if source_type == 'solar':
            # 太阳能: 光照强度 (lux) -> 功率
            lux = conditions.get('light', 0)
            efficiency = 0.15  # 15%效率
            area = 0.001  # 1 cm²
            self.energy_harvesting_rate = lux * efficiency * area * 0.001  # mW

        elif source_type == 'thermal':
            # 热能: 温差 -> 功率
            temp_diff = conditions.get('temp_diff', 0)
            efficiency = 0.05
            self.energy_harvesting_rate = temp_diff * efficiency * 10  # mW

        elif source_type == 'vibration':
            # 振动能
            acceleration = conditions.get('acceleration', 0)
            self.energy_harvesting_rate = acceleration ** 2 * 0.1  # mW

        elif source_type == 'rf':
            # RF能量收集
            rssi = conditions.get('rssi', -100)  # dBm
            if rssi > -50:
                self.energy_harvesting_rate = (rssi + 100) * 0.01  # mW
            else:
                self.energy_harvesting_rate = 0.0

        return self.energy_harvesting_rate

    def predict_battery_lifetime(self, usage_profile: Dict) -> Dict:
        """
        预测电池寿命

        Args:
            usage_profile: 使用模式
                {
                    'active_hours_per_day': 8,
                    'transmissions_per_day': 24,
                    'energy_harvesting': True
                }

        Returns:
            寿命预测结果
        """
        active_hours = usage_profile.get('active_hours_per_day', 8)
        daily_tx = usage_profile.get('transmissions_per_day', 24)
        harvesting = usage_profile.get('energy_harvesting', False)

        # 计算每日能耗
        sleep_hours = 24 - active_hours

        sleep_energy = sleep_hours * 3600 * self.consumption_model['sleep'] / 1000  # mWh
        active_energy = active_hours * 3600 * self.consumption_model['active'] / 1000  # mWh
        tx_energy = daily_tx * self.consumption_model['transmit'] * 2 / 1000  # 假设2秒传输

        daily_consumption = sleep_energy + active_energy + tx_energy

        # 能量收集贡献
        daily_harvesting = 0
        if harvesting:
            # 假设每天8小时有效收集
            daily_harvesting = self.energy_harvesting_rate * 8  # mWh

        net_daily_consumption = daily_consumption - daily_harvesting

        # 总可用能量
        total_energy = self.current_battery * 3.7  # mWh

        # 预测寿命
        if net_daily_consumption <= 0:
            lifetime_days = float('inf')
            lifetime_message = "Unlimited (energy harvesting exceeds consumption)"
        else:
            lifetime_days = total_energy / net_daily_consumption
            lifetime_message = f"{lifetime_days:.1f} days"

        return {
            'current_battery_mah': self.current_battery,
            'daily_consumption_mwh': daily_consumption,
            'daily_harvesting_mwh': daily_harvesting,
            'net_daily_consumption_mwh': net_daily_consumption,
            'predicted_lifetime_days': lifetime_days if lifetime_days != float('inf') else -1,
            'predicted_lifetime_message': lifetime_message,
            'battery_end_of_life': lifetime_days < 365 if lifetime_days != float('inf') else False
        }

    def optimize_for_lifetime(self, target_days: float) -> Dict:
        """
        优化配置以达到目标寿命

        Args:
            target_days: 目标运行天数

        Returns:
            优化建议
        """
        recommendations = []

        # 计算所需占空比
        optimal_duty = self.calculate_duty_cycle_for_lifetime(target_days)
        if optimal_duty < self.duty_cycle:
            recommendations.append({
                'parameter': 'duty_cycle',
                'current': self.duty_cycle,
                'recommended': optimal_duty,
                'impact': 'Reduce power consumption during sleep'
            })

        # 检查采样率
        if self.sampling_rate > 0.5:
            recommendations.append({
                'parameter': 'sampling_rate',
                'current': self.sampling_rate,
                'recommended': 0.5,
                'impact': 'Reduce sensor sampling frequency'
            })

        # 能量收集建议
        if self.energy_harvesting_rate < 1.0:
            recommendations.append({
                'parameter': 'energy_harvesting',
                'current': self.energy_harvesting_rate,
                'recommended': 'Implement solar or thermal harvesting',
                'impact': 'Extend battery life significantly'
            })

        return {
            'target_days': target_days,
            'recommendations': recommendations,
            'estimated_savings_percent': len(recommendations) * 15
        }


# =============================================================================
# IoT平台 (IoTPlatform)
# =============================================================================

class IoTPlatform:
    """
    完整IoT平台模拟
    支持设备管理、规则引擎、告警系统和仪表板指标
    """

    def __init__(self, platform_id: str, config: IoTConfig):
        self.platform_id = platform_id
        self.config = config
        self.devices: Dict[str, IoTDevice] = {}
        self.gateways: Dict[str, IoTGateway] = {}
        self.database = TimeSeriesDatabase(config)
        self.anomaly_detector = AnomalyDetection(config)
        self.security = IoTSecurity(config)
        self.rules: List[Dict] = []
        self.alerts: deque = deque(maxlen=1000)
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None

    # ============= 设备管理 =============

    def register_device(self, device: IoTDevice) -> bool:
        """注册设备"""
        if device.device_id in self.devices:
            return False

        self.devices[device.device_id] = device
        return True

    def unregister_device(self, device_id: str):
        """注销设备"""
        if device_id in self.devices:
            del self.devices[device_id]

    def get_device_status(self, device_id: str) -> Optional[Dict]:
        """获取设备状态"""
        device = self.devices.get(device_id)
        return device.get_status() if device else None

    def list_devices(self) -> List[Dict]:
        """列出所有设备"""
        return [device.get_status() for device in self.devices.values()]

    # ============= 规则引擎 =============

    def add_rule(self, rule_id: str, condition: Callable, action: Callable,
                 description: str = ""):
        """
        添加规则

        Args:
            rule_id: 规则ID
            condition: 条件函数 (data) -> bool
            action: 动作函数 (data) -> None
            description: 规则描述
        """
        self.rules.append({
            'id': rule_id,
            'condition': condition,
            'action': action,
            'description': description,
            'enabled': True,
            'trigger_count': 0
        })

    def evaluate_rules(self, data: Dict):
        """评估所有规则"""
        for rule in self.rules:
            if not rule['enabled']:
                continue

            try:
                if rule['condition'](data):
                    rule['action'](data)
                    rule['trigger_count'] += 1
            except Exception as e:
                print(f"Rule {rule['id']} evaluation error: {e}")

    def enable_rule(self, rule_id: str):
        """启用规则"""
        for rule in self.rules:
            if rule['id'] == rule_id:
                rule['enabled'] = True
                break

    def disable_rule(self, rule_id: str):
        """禁用规则"""
        for rule in self.rules:
            if rule['id'] == rule_id:
                rule['enabled'] = False
                break

    # ============= 告警系统 =============

    def create_alert(self, alert_type: str, severity: str, message: str,
                     source: str, metadata: Dict = None):
        """创建告警"""
        alert = {
            'id': hashlib.sha256(f"{time.time()}:{message}".encode()).hexdigest()[:16],
            'timestamp': time.time(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'source': source,
            'metadata': metadata or {},
            'acknowledged': False
        }

        self.alerts.append(alert)

        # 更新指标
        self.metrics['alerts'].append({
            'timestamp': alert['timestamp'],
            'type': alert_type,
            'severity': severity
        })

        return alert['id']

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                return True
        return False

    def get_active_alerts(self, severity_filter: str = None) -> List[Dict]:
        """获取活动告警"""
        alerts = [a for a in self.alerts if not a['acknowledged']]
        if severity_filter:
            alerts = [a for a in alerts if a['severity'] == severity_filter]
        return alerts

    # ============= 仪表板指标 =============

    def record_metric(self, metric_name: str, value: float, tags: Dict = None):
        """记录指标"""
        self.metrics[metric_name].append({
            'timestamp': time.time(),
            'value': value,
            'tags': tags or {}
        })

    def get_metric_stats(self, metric_name: str, time_range: float = 3600) -> Dict:
        """获取指标统计"""
        current_time = time.time()
        values = [
            m['value'] for m in self.metrics[metric_name]
            if current_time - m['timestamp'] <= time_range
        ]

        if not values:
            return {'count': 0}

        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'current': values[-1] if values else None
        }

    def get_dashboard_data(self) -> Dict:
        """获取仪表板数据"""
        return {
            'platform_id': self.platform_id,
            'timestamp': time.time(),
            'device_count': len(self.devices),
            'online_devices': sum(1 for d in self.devices.values() if d.connected),
            'active_alerts': len(self.get_active_alerts()),
            'metrics': {
                name: self.get_metric_stats(name)
                for name in self.metrics.keys()
            },
            'recent_alerts': list(self.alerts)[-10:]
        }

    # ============= 数据处理 =============

    def process_data(self, device_id: str, data: Dict):
        """处理设备数据"""
        # 存储到时序数据库
        for reading in data.get('data', []):
            series_id = f"{device_id}:{reading.get('sensor_type', 'unknown')}"
            self.database.write(
                series_id,
                reading.get('timestamp', time.time()),
                reading.get('value', 0),
                reading.get('tags', {})
            )

        # 异常检测
        for reading in data.get('data', []):
            series_id = f"{device_id}:{reading.get('sensor_type', 'unknown')}"
            is_anomaly, score = self.anomaly_detector.detect_with_lstm(
                series_id,
                reading.get('value', 0)
            )

            if is_anomaly:
                self.create_alert(
                    'anomaly',
                    'warning',
                    f"Anomaly detected in {series_id}: score={score:.2f}",
                    device_id,
                    {'anomaly_score': score, 'value': reading.get('value')}
                )

        # 规则引擎
        self.evaluate_rules(data)

        # 更新指标
        self.record_metric('data_points_processed', len(data.get('data', [])))

    def start(self):
        """启动平台"""
        self._running = True
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()

    def stop(self):
        """停止平台"""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

    def _processing_loop(self):
        """数据处理循环"""
        while self._running:
            # 处理设备数据
            for device in self.devices.values():
                while device.message_queue:
                    try:
                        message = device.message_queue.popleft()
                        self.process_data(device.device_id, message['packet'])
                    except IndexError:
                        break

            time.sleep(0.1)


# =============================================================================
# 演示和测试
# =============================================================================

def demo_iot_system():
    """IoT系统演示"""
    print("=" * 60)
    print("IoT物联网系统综合演示")
    print("=" * 60)

    # 创建配置
    config = IoTConfig()
    print("\n[1] 配置管理")
    print(f"  默认采样率: {config.default_sampling_rate} Hz")
    print(f"  电池容量: {config.default_battery_capacity} mAh")

    # 创建传感器
    print("\n[2] 传感器模拟")
    temp_sensor = Sensor(SensorType.TEMPERATURE, config)
    readings = [temp_sensor.read(time.time() + i) for i in range(5)]
    print(f"  温度传感器生成 {len(readings)} 个读数")
    for r in readings[:3]:
        print(f"    时间: {r['timestamp']:.2f}, 值: {r['value']:.2f}{r['unit']}, 质量: {r['quality']:.2f}")

    # 数据压缩演示
    values = [r['value'] for r in readings]
    encoded = Sensor.delta_encode(values)
    decoded = Sensor.delta_decode(encoded)
    print(f"  Delta编码: {values} -> {encoded}")
    print(f"  解码验证: {decoded}")

    # 创建设备
    print("\n[3] IoT设备模拟")
    device = IoTDevice("dev_001", config)
    device.add_sensor(SensorType.TEMPERATURE)
    device.add_sensor(SensorType.HUMIDITY)
    device.connect()
    print(f"  设备 {device.device_id} 已创建")
    print(f"  传感器: {[s.name for s in device.sensors.keys()]}")
    print(f"  连接状态: {device.connected}")

    # 创建网关
    print("\n[4] IoT网关")
    gateway = IoTGateway("gw_001", config)
    gateway.register_device(device)
    print(f"  网关 {gateway.gateway_id} 已创建")
    print(f"  注册设备数: {len(gateway.devices)}")

    # 流处理
    print("\n[5] 流数据处理")
    streaming = DataStreaming(config)
    streaming.create_window("temp_window", 10)
    for i, r in enumerate(readings):
        streaming.add_to_window("temp_window", r)
    avg = streaming.sliding_window_operation("temp_window", "avg")
    print(f"  滑动窗口平均值: {avg:.2f}")

    # 异常检测
    print("\n[6] 异常检测")
    detector = AnomalyDetection(config)
    test_data = [20, 21, 20.5, 21.5, 20, 50, 21, 20.5]  # 50是异常值
    anomalies = detector.zscore_detection(test_data)
    print(f"  测试数据: {test_data}")
    print(f"  Z-Score检测到的异常: {anomalies}")

    ensemble_result = detector.ensemble_detection(test_data)
    print(f"  集成检测结果: 异常={ensemble_result['anomaly']}, 置信度={ensemble_result['confidence']:.2f}")

    # 时序数据库
    print("\n[7] 时序数据库")
    tsdb = TimeSeriesDatabase(config)
    now = time.time()
    for i in range(10):
        tsdb.write("sensor_001", now + i, 20 + i * 0.5)
    stats = tsdb.get_statistics("sensor_001")
    print(f"  写入10个数据点")
    print(f"  统计: 数量={stats['count']}, 平均值={stats['avg']:.2f}")

    # 数字孪生
    print("\n[8] 数字孪生")
    twin = DigitalTwin("twin_001", config)
    twin.define_physical_model({
        'type': 'thermal',
        'thermal_mass': 1000,
        'heat_transfer_coeff': 10,
        'ambient_temp': 25
    })
    twin.update_state({'temperature': 30})
    simulation = twin.simulate(10, 1.0)
    print(f"  数字孪生 {twin.twin_id} 已创建")
    print(f"  热力学模型仿真 {len(simulation)} 步")
    print(f"  最终温度: {simulation[-1]['state']['temperature']:.2f}°C")

    # 安全机制
    print("\n[9] IoT安全")
    security = IoTSecurity(config)
    token = security.generate_token("user_001", ["read", "write"])
    verified = security.verify_token(token)
    print(f"  生成令牌: {token[:16]}...")
    print(f"  令牌验证: {'成功' if verified else '失败'}")

    # 加密演示
    key = b"secret_key_12345"
    message = b"Hello IoT"
    encrypted = security.xor_encrypt(message, key)
    decrypted = security.xor_decrypt(encrypted, key)
    print(f"  XOR加密: {message} -> {encrypted.hex()}")
    print(f"  解密验证: {decrypted}")

    # 能源管理
    print("\n[10] 能源管理")
    energy = EnergyManagement(config)
    lifetime = energy.predict_battery_lifetime({
        'active_hours_per_day': 8,
        'transmissions_per_day': 24,
        'energy_harvesting': False
    })
    print(f"  电池寿命预测: {lifetime['predicted_lifetime_message']}")
    print(f"  每日能耗: {lifetime['daily_consumption_mwh']:.2f} mWh")

    # IoT平台
    print("\n[11] IoT平台")
    platform = IoTPlatform("platform_001", config)
    platform.register_device(device)

    # 添加规则
    def high_temp_condition(data):
        for reading in data.get('data', []):
            if reading.get('sensor_type') == 'TEMPERATURE' and reading.get('value', 0) > 30:
                return True
        return False

    def high_temp_action(data):
        print(f"    [规则触发] 高温告警!")

    platform.add_rule("high_temp_rule", high_temp_condition, high_temp_action, "Temperature > 30C")

    # 模拟数据触发规则
    test_packet = {
        'device_id': 'dev_001',
        'data': [{'sensor_type': 'TEMPERATURE', 'value': 35, 'timestamp': time.time()}]
    }
    platform.evaluate_rules(test_packet)

    dashboard = platform.get_dashboard_data()
    print(f"  平台 {platform.platform_id} 已启动")
    print(f"  设备数量: {dashboard['device_count']}")
    print(f"  在线设备: {dashboard['online_devices']}")
    print(f"  活动告警: {dashboard['active_alerts']}")

    print("\n" + "=" * 60)
    print("IoT系统演示完成!")
    print("=" * 60)

    return {
        'config': config,
        'device': device,
        'gateway': gateway,
        'platform': platform,
        'detector': detector,
        'twin': twin,
        'security': security,
        'energy': energy
    }


if __name__ == "__main__":
    demo_iot_system()
