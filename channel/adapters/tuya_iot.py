"""
涂鸦智能 IoT 集成适配器模块

基于涂鸦开放平台 (Tuya Open Platform) 的 IoT 设备管理适配器，
支持设备控制、场景管理、房间管理、自动化规则和能源监控。

支持的 API 版本: v2.0
官方文档: https://developer.tuya.com/en/docs/iot/open-api/api-list

Author: AGI Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import aiohttp

from ...base import (
    ChannelAdapter,
    ChannelCapability,
    ChannelConfig,
    ConnectionState,
    MessagePriority,
    ReceiveResult,
    SendResult,
)
from ...universal_message import UniversalMessage

logger = logging.getLogger(__name__)


# ============================================================
# 异常定义
# ============================================================

class TuyaError(Exception):
    """涂鸦平台异常基类"""

    def __init__(
        self,
        code: int,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[TuyaError {code}] {message}")

    @classmethod
    def from_response(cls, body: Dict[str, Any]) -> TuyaError:
        """从 API 响应构造异常"""
        return cls(
            code=body.get("code", -1),
            message=body.get("msg", "未知错误"),
            details=body,
        )

    @property
    def is_auth_error(self) -> bool:
        """是否为鉴权错误"""
        return self.code in (1004, 1010, 1106, 2009, 2100, 28841003, 28841004)

    @property
    def is_rate_limited(self) -> bool:
        """是否被限流"""
        return self.code == 28841002

    @property
    def is_retryable(self) -> bool:
        """是否可重试"""
        return self.is_rate_limited or self.code in (2008, 2009, 2014)


# ============================================================
# 配置
# ============================================================

@dataclass
class TuyaConfig(ChannelConfig):
    """涂鸦 IoT 适配器配置

    Attributes:
        access_id: 涂鸦开放平台 Access ID
        access_key: 涂鸦开放平台 Access Key (Secret)
        endpoint: API 端点地址，默认中国区
        mqtt_broker: MQTT 消息代理地址
        schema: 数据中心标识，默认中国区
    """

    access_id: str = ""
    access_key: str = ""
    endpoint: str = "https://openapi.tuyacn.com"
    mqtt_broker: str = "mqtts://m1.tuyacn.com:8883"
    schema: str = ""  # 留空自动根据 endpoint 推断


# ============================================================
# 数据模型
# ============================================================

class DeviceStatus(str, Enum):
    """设备在线状态"""
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass
class TuyaDevice:
    """涂鸦设备信息"""
    device_id: str
    name: str
    model: str = ""
    category: str = ""
    status: DeviceStatus = DeviceStatus.OFFLINE
    online: bool = False
    ip: str = ""
    uid: str = ""
    local_key: str = ""
    sub: bool = False
    uuid: str = ""
    owner_id: str = ""
    product_id: str = ""
    product_name: str = ""
    icon: str = ""
    capabilities: List[str] = field(default_factory=list)
    dps: Dict[str, Any] = field(default_factory=dict)  # 数据点
    room_id: int = 0
    home_id: int = 0

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> TuyaDevice:
        """从 API 响应构造设备对象"""
        return cls(
            device_id=data.get("id", ""),
            name=data.get("name", ""),
            model=data.get("model", ""),
            category=data.get("category", ""),
            online=data.get("online", False),
            status=DeviceStatus.ONLINE if data.get("online") else DeviceStatus.OFFLINE,
            ip=data.get("ip", ""),
            uid=data.get("uid", ""),
            local_key=data.get("local_key", ""),
            sub=data.get("sub", False),
            uuid=data.get("uuid", ""),
            owner_id=data.get("owner_id", ""),
            product_id=data.get("product_id", ""),
            product_name=data.get("product_name", ""),
            icon=data.get("icon", ""),
            room_id=data.get("room_id", 0),
            home_id=data.get("home_id", 0),
        )


@dataclass
class TuyaScene:
    """涂鸦场景信息"""
    scene_id: str
    name: str
    status: int = 1  # 1=启用, 0=禁用
    actions: List[Dict[str, Any]] = field(default_factory=list)
    preconditions: List[Dict[str, Any]] = field(default_factory=list)
    home_id: int = 0
    created_time: str = ""

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> TuyaScene:
        return cls(
            scene_id=data.get("scene_id", data.get("id", "")),
            name=data.get("name", ""),
            status=data.get("status", 1),
            actions=data.get("action_list", data.get("actions", [])),
            preconditions=data.get("condition_list", data.get("preconditions", [])),
            home_id=data.get("home_id", 0),
            created_time=data.get("created_time", ""),
        )


@dataclass
class TuyaRoom:
    """涂鸦房间信息"""
    room_id: int
    name: str
    home_id: int = 0
    device_count: int = 0
    background: str = ""

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> TuyaRoom:
        return cls(
            room_id=data.get("room_id", 0),
            name=data.get("name", ""),
            home_id=data.get("home_id", 0),
            device_count=data.get("device_count", 0),
            background=data.get("background", ""),
        )


@dataclass
class TuyaAutomationRule:
    """涂鸦自动化规则"""
    rule_id: str
    name: str
    status: int = 1
    enabled: bool = True
    actions: List[Dict[str, Any]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    match_type: int = 1  # 1=全部满足, 2=任一满足
    created_time: str = ""

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> TuyaAutomationRule:
        return cls(
            rule_id=data.get("rule_id", data.get("id", "")),
            name=data.get("name", ""),
            status=data.get("status", 1),
            enabled=data.get("status", 1) == 1,
            actions=data.get("action_list", []),
            conditions=data.get("condition_list", []),
            match_type=data.get("match_type", 1),
            created_time=data.get("created_time", ""),
        )


@dataclass
class EnergyData:
    """能源监控数据"""
    device_id: str
    stat_date: str
    stat_type: str  # day / month
    kwh: float = 0.0
    kwh_day: float = 0.0
    kwh_night: float = 0.0
    currency: str = "CNY"
    cost: float = 0.0

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> EnergyData:
        return cls(
            device_id=data.get("device_id", ""),
            stat_date=data.get("stat_date", ""),
            stat_type=data.get("stat_type", "day"),
            kwh=data.get("kwh", 0.0),
            kwh_day=data.get("kwh_day", 0.0),
            kwh_night=data.get("kwh_night", 0.0),
            currency=data.get("currency", "CNY"),
            cost=data.get("cost", 0.0),
        )


# ============================================================
# 签名工具
# ============================================================

class TuyaSigner:
    """涂鸦 API HMAC-SHA256 签名工具"""

    @staticmethod
    def sign(
        method: str,
        path: str,
        headers: Dict[str, str],
        query: str,
        body: str,
        access_key: str,
        timestamp: str,
    ) -> str:
        """生成 HMAC-SHA256 签名

        签名规则:
            string_to_sign = method + "\\n" + sign_headers + "\\n" + sign_url + "\\n" + sign_body + "\\n" + timestamp
            sign = HMAC-SHA256(access_key, string_to_sign).hexdigest()
        """
        # 按字典序排列需要签名的 header
        sign_header_keys = sorted(
            k for k in headers if k.lower() in ("client_id", "sign_timestamp", "nonce")
        )
        sign_headers = "\n".join(f"{k}:{headers[k]}" for k in sign_header_keys)

        # 构造待签名字符串
        str_to_sign = "\n".join([
            method.upper(),
            sign_headers,
            urllib.parse.quote(path, safe="") + "?" + query if query else urllib.parse.quote(path, safe=""),
            hashlib.md5(body.encode("utf-8")).hexdigest() if body else hashlib.md5(b"").hexdigest(),
            timestamp,
        ])

        # HMAC-SHA256
        signature = hmac.new(
            access_key.encode("utf-8"),
            str_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        return signature.upper()


# ============================================================
# MQTT 消息处理
# ============================================================

@dataclass
class MQTTMessage:
    """MQTT 消息封装"""
    topic: str
    payload: bytes
    qos: int = 0
    retain: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def decoded(self) -> Dict[str, Any]:
        """解码 payload 为字典"""
        try:
            return json.loads(self.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"raw": self.payload.hex()}


class DeviceStatusCallback:
    """设备状态变更回调管理"""

    def __init__(self) -> None:
        self._callbacks: List[Callable[[str, Dict[str, Any]], Any]] = []

    def register(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        """注册设备状态变更回调"""
        self._callbacks.append(callback)

    def unregister(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        """注销回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def dispatch(self, device_id: str, status: Dict[str, Any]) -> None:
        """分发设备状态变更事件"""
        for cb in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(device_id, status)
                else:
                    cb(device_id, status)
            except Exception as exc:
                logger.error("设备状态回调执行失败 (device=%s): %s", device_id, exc)


# ============================================================
# 主适配器
# ============================================================

class TuyaIoTAdapter(ChannelAdapter):
    """涂鸦智能 IoT 集成适配器

    功能:
        - 设备管理: 列出设备、查询状态、控制设备
        - 场景管理: 列出场景、激活场景
        - 房间管理: 列出房间、查询房间内设备
        - 自动化规则: 列出规则、启用/禁用
        - 实时设备状态: MQTT 订阅
        - 能源监控: 查询设备功耗数据

    Example:
        config = TuyaConfig(channel_id="tuya", access_id="xxx", access_key="yyy")
        adapter = TuyaIoTAdapter(config)
        await adapter.connect()
        devices = await adapter.list_devices()
    """

    # API 路径常量
    _TOKEN_PATH = "/v1.0/token"
    _DEVICE_LIST_PATH = "/v2.0/cloud/devices"
    _DEVICE_STATUS_PATH = "/v1.0/iot-03/devices/{device_id}/status"
    _DEVICE_CONTROL_PATH = "/v1.0/iot-03/devices/{device_id}/commands"
    _SCENE_LIST_PATH = "/v1.0/scenes"
    _SCENE_EXECUTE_PATH = "/v1.0/scenes/{scene_id}/execute"
    _ROOM_LIST_PATH = "/v2.0/cloud/family/{family_id}/rooms"
    _ROOM_DEVICES_PATH = "/v2.0/cloud/family/{family_id}/rooms/{room_id}/devices"
    _AUTOMATION_LIST_PATH = "/v1.0/automations/rules"
    _AUTOMATION_STATUS_PATH = "/v1.0/automations/rules/{rule_id}/actions/enable"
    _ENERGY_DAY_PATH = "/v1.0/iot-03/devices/{device_id}/energy-consumption/day"
    _ENERGY_MONTH_PATH = "/v1.0/iot-03/devices/{device_id}/energy-consumption/month"

    def __init__(self, config: TuyaConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._refresh_token: Optional[str] = None
        self._mqtt_client: Optional[aiohttp.ClientSession] = None
        self._status_callbacks = DeviceStatusCallback()
        self._device_cache: Dict[str, TuyaDevice] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    # ----------------------------------------------------------
    # 能力声明
    # ----------------------------------------------------------

    def _initialize_capabilities(self) -> None:
        self._capabilities = {
            ChannelCapability.WEBHOOK_MODE,
            ChannelCapability.CHANNEL_INFO,
            ChannelCapability.USER_INFO,
            ChannelCapability.RATE_LIMITING,
        }

    # ----------------------------------------------------------
    # 连接生命周期
    # ----------------------------------------------------------

    async def _connect_impl(self) -> bool:
        """连接涂鸦平台，获取 access_token"""
        try:
            await self._fetch_token()
            self._logger.info("成功连接涂鸦平台, token 前缀: %s...", self._access_token[:16] if self._access_token else "")
            return True
        except Exception as exc:
            self._logger.error("连接涂鸦平台失败: %s", exc)
            return False

    async def _disconnect_impl(self) -> None:
        """断开连接，清理资源"""
        self._access_token = None
        self._token_expires_at = 0.0
        self._refresh_token = None
        self._device_cache.clear()
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._logger.info("已断开涂鸦平台连接")

    async def _health_check_impl(self) -> bool:
        """健康检查: 验证 token 是否有效"""
        try:
            if not self._access_token or time.time() >= self._token_expires_at:
                await self._fetch_token()
            return self._access_token is not None
        except Exception:
            return False

    # ----------------------------------------------------------
    # Token 管理
    # ----------------------------------------------------------

    async def _fetch_token(self, force: bool = False) -> str:
        """获取或刷新 access_token

        使用 HMAC-SHA256 签名方式获取 token，token 有效期为 86400 秒。
        """
        now = time.time()
        if not force and self._access_token and now < self._token_expires_at - 300:
            return self._access_token

        timestamp = str(int(now * 1000))
        nonce = hashlib.md5(timestamp.encode()).hexdigest()

        path = self._TOKEN_PATH
        query = ""
        body = ""

        headers = {
            "client_id": self._cfg.access_id,
            "sign_timestamp": timestamp,
            "nonce": nonce,
            "sign_method": "HMAC-SHA256",
            "t": timestamp,
            "Signature-Headers": "client_id",
        }

        sign = TuyaSigner.sign(
            method="GET",
            path=path,
            headers=headers,
            query=query,
            body=body,
            access_key=self._cfg.access_key,
            timestamp=timestamp,
        )
        headers["sign"] = sign

        url = self._cfg.endpoint + path
        async with self._get_session() as session:
            async with session.get(url, headers=headers) as resp:
                result = await resp.json()

        if result.get("success") is not True:
            raise TuyaError.from_response(result)

        token_data = result.get("result", {})
        self._access_token = token_data.get("access_token", "")
        self._refresh_token = token_data.get("refresh_token", "")
        expire_in = token_data.get("expire_time", 86400)
        self._token_expires_at = now + expire_in - 300  # 提前 5 分钟刷新

        return self._access_token

    # ----------------------------------------------------------
    # HTTP 请求封装
    # ----------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp 会话"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送签名 API 请求

        自动附加签名头和 access_token，处理 token 过期自动刷新。
        """
        await self._fetch_token()

        timestamp = str(int(time.time() * 1000))
        nonce = hashlib.md5(timestamp.encode()).hexdigest()
        body_str = json.dumps(body, ensure_ascii=False) if body else ""
        query_str = urllib.parse.urlencode(sorted(params.items())) if params else ""

        headers = {
            "client_id": self._cfg.access_id,
            "access_token": self._access_token,
            "sign_timestamp": timestamp,
            "nonce": nonce,
            "sign_method": "HMAC-SHA256",
            "t": timestamp,
            "Signature-Headers": "client_id",
            "Content-Type": "application/json",
        }

        sign = TuyaSigner.sign(
            method=method,
            path=path,
            headers=headers,
            query=query_str,
            body=body_str,
            access_key=self._cfg.access_key,
            timestamp=timestamp,
        )
        headers["sign"] = sign

        url = self._cfg.endpoint + path
        if query_str:
            url += "?" + query_str

        async with self._get_session() as session:
            if method.upper() == "GET":
                async with session.get(url, headers=headers) as resp:
                    result = await resp.json()
            elif method.upper() == "POST":
                async with session.post(url, headers=headers, data=body_str.encode("utf-8")) as resp:
                    result = await resp.json()
            elif method.upper() == "PUT":
                async with session.put(url, headers=headers, data=body_str.encode("utf-8")) as resp:
                    result = await resp.json()
            elif method.upper() == "DELETE":
                async with session.delete(url, headers=headers) as resp:
                    result = await resp.json()
            else:
                raise ValueError(f"不支持的 HTTP 方法: {method}")

        if result.get("success") is not True:
            # token 过期则自动刷新重试一次
            if result.get("code") in (1010, 28841003):
                self._logger.warning("Token 过期，正在刷新重试...")
                await self._fetch_token(force=True)
                return await self._request(method, path, params, body)
            raise TuyaError.from_response(result)

        return result

    # ----------------------------------------------------------
    # 设备管理
    # ----------------------------------------------------------

    async def list_devices(
        self,
        page_no: int = 1,
        page_size: int = 50,
        device_ids: Optional[List[str]] = None,
    ) -> Tuple[List[TuyaDevice], int]:
        """获取设备列表

        Args:
            page_no: 页码，从 1 开始
            page_size: 每页数量，最大 100
            device_ids: 可选，按设备 ID 过滤

        Returns:
            (设备列表, 总数)
        """
        params: Dict[str, Any] = {"page_no": page_no, "page_size": page_size}
        if device_ids:
            params["device_ids"] = ",".join(device_ids)

        result = await self._request("GET", self._DEVICE_LIST_PATH, params=params)
        data = result.get("result", {})

        devices = [TuyaDevice.from_api(d) for d in data.get("list", [])]
        total = data.get("total", 0)

        # 更新缓存
        for dev in devices:
            self._device_cache[dev.device_id] = dev

        return devices, total

    async def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """获取设备实时状态（所有数据点）

        Args:
            device_id: 设备 ID

        Returns:
            设备状态字典，包含各数据点的值
        """
        path = self._DEVICE_STATUS_PATH.format(device_id=device_id)
        result = await self._request("GET", path)
        return result.get("result", {})

    async def control_device(
        self,
        device_id: str,
        commands: List[Dict[str, Any]],
    ) -> bool:
        """控制设备（下发指令）

        Args:
            device_id: 设备 ID
            commands: 指令列表，每条格式 {"code": "switch_1", "value": true}

        Returns:
            是否成功

        Example:
            await adapter.control_device("dev_xxx", [
                {"code": "switch_1", "value": True},
                {"code": "brightness", "value": 50},
            ])
        """
        path = self._DEVICE_CONTROL_PATH.format(device_id=device_id)
        result = await self._request("POST", path, body={"commands": commands})
        return result.get("success", False)

    async def turn_on_device(self, device_id: str, switch_code: str = "switch_1") -> bool:
        """打开设备

        Args:
            device_id: 设备 ID
            switch_code: 开关数据点编码，默认 switch_1

        Returns:
            是否成功
        """
        return await self.control_device(device_id, [{"code": switch_code, "value": True}])

    async def turn_off_device(self, device_id: str, switch_code: str = "switch_1") -> bool:
        """关闭设备

        Args:
            device_id: 设备 ID
            switch_code: 开关数据点编码，默认 switch_1

        Returns:
            是否成功
        """
        return await self.control_device(device_id, [{"code": switch_code, "value": False}])

    async def set_device_value(
        self,
        device_id: str,
        code: str,
        value: Any,
    ) -> bool:
        """设置设备数据点值

        Args:
            device_id: 设备 ID
            code: 数据点编码
            value: 目标值

        Returns:
            是否成功
        """
        return await self.control_device(device_id, [{"code": code, "value": value}])

    # ----------------------------------------------------------
    # 场景管理
    # ----------------------------------------------------------

    async def list_scenes(self, home_id: Optional[int] = None) -> List[TuyaScene]:
        """获取场景列表

        Args:
            home_id: 家庭 ID，不传则返回所有

        Returns:
            场景列表
        """
        params: Dict[str, Any] = {}
        if home_id is not None:
            params["home_id"] = home_id

        result = await self._request("GET", self._SCENE_LIST_PATH, params=params)
        scenes = result.get("result", [])
        return [TuyaScene.from_api(s) for s in scenes]

    async def activate_scene(self, scene_id: str) -> bool:
        """激活（执行）场景

        Args:
            scene_id: 场景 ID

        Returns:
            是否成功
        """
        path = self._SCENE_EXECUTE_PATH.format(scene_id=scene_id)
        result = await self._request("POST", path)
        return result.get("success", False)

    # ----------------------------------------------------------
    # 房间管理
    # ----------------------------------------------------------

    async def list_rooms(self, family_id: int) -> List[TuyaRoom]:
        """获取房间列表

        Args:
            family_id: 家庭 ID

        Returns:
            房间列表
        """
        path = self._ROOM_LIST_PATH.format(family_id=family_id)
        result = await self._request("GET", path)
        rooms = result.get("result", [])
        return [TuyaRoom.from_api(r) for r in rooms]

    async def get_room_devices(
        self,
        family_id: int,
        room_id: int,
        page_no: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[TuyaDevice], int]:
        """获取房间内的设备列表

        Args:
            family_id: 家庭 ID
            room_id: 房间 ID
            page_no: 页码
            page_size: 每页数量

        Returns:
            (设备列表, 总数)
        """
        path = self._ROOM_DEVICES_PATH.format(family_id=family_id, room_id=room_id)
        params = {"page_no": page_no, "page_size": page_size}
        result = await self._request("GET", path, params=params)
        data = result.get("result", {})
        devices = [TuyaDevice.from_api(d) for d in data.get("list", [])]
        total = data.get("total", 0)
        return devices, total

    # ----------------------------------------------------------
    # 自动化规则
    # ----------------------------------------------------------

    async def list_automation_rules(self, home_id: Optional[int] = None) -> List[TuyaAutomationRule]:
        """获取自动化规则列表

        Args:
            home_id: 家庭 ID

        Returns:
            规则列表
        """
        params: Dict[str, Any] = {}
        if home_id is not None:
            params["home_id"] = home_id

        result = await self._request("GET", self._AUTOMATION_LIST_PATH, params=params)
        rules = result.get("result", [])
        return [TuyaAutomationRule.from_api(r) for r in rules]

    async def set_automation_enabled(self, rule_id: str, enabled: bool) -> bool:
        """启用或禁用自动化规则

        Args:
            rule_id: 规则 ID
            enabled: True 启用，False 禁用

        Returns:
            是否成功
        """
        path = self._AUTOMATION_STATUS_PATH.format(rule_id=rule_id)
        result = await self._request("PUT", path, body={"enable": enabled})
        return result.get("success", False)

    # ----------------------------------------------------------
    # 能源监控
    # ----------------------------------------------------------

    async def get_energy_day(
        self,
        device_id: str,
        start_day: str,
        end_day: str,
    ) -> List[EnergyData]:
        """获取设备每日能耗数据

        Args:
            device_id: 设备 ID
            start_day: 起始日期，格式 YYYYMMDD
            end_day: 结束日期，格式 YYYYMMDD

        Returns:
            每日能耗数据列表
        """
        path = self._ENERGY_DAY_PATH.format(device_id=device_id)
        params = {"start_day": start_day, "end_day": end_day}
        result = await self._request("GET", path, params=params)
        items = result.get("result", [])
        return [EnergyData.from_api(item) for item in items]

    async def get_energy_month(
        self,
        device_id: str,
        start_month: str,
        end_month: str,
    ) -> List[EnergyData]:
        """获取设备每月能耗数据

        Args:
            device_id: 设备 ID
            start_month: 起始月份，格式 YYYYMM
            end_month: 结束月份，格式 YYYYMM

        Returns:
            每月能耗数据列表
        """
        path = self._ENERGY_MONTH_PATH.format(device_id=device_id)
        params = {"start_month": start_month, "end_month": end_month}
        result = await self._request("GET", path, params=params)
        items = result.get("result", [])
        return [EnergyData.from_api(item) for item in items]

    # ----------------------------------------------------------
    # MQTT 实时设备状态
    # ----------------------------------------------------------

    def register_status_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], Any],
    ) -> None:
        """注册设备状态变更回调

        Args:
            callback: 回调函数，接收 (device_id, status_dict)
        """
        self._status_callbacks.register(callback)

    def unregister_status_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], Any],
    ) -> None:
        """注销设备状态变更回调"""
        self._status_callbacks.unregister(callback)

    async def _handle_mqtt_message(self, message: MQTTMessage) -> None:
        """处理收到的 MQTT 消息"""
        try:
            payload = message.decoded
            device_id = payload.get("devId", payload.get("device_id", ""))
            data = payload.get("data", payload)
            if device_id:
                await self._status_callbacks.dispatch(device_id, data)
                self._logger.debug("收到设备状态变更: device=%s", device_id)
        except Exception as exc:
            self._logger.error("处理 MQTT 消息失败: %s", exc)

    async def subscribe_device_status(self, device_ids: List[str]) -> None:
        """订阅设备状态变更

        通过涂鸦消息转发服务或 MQTT 订阅设备状态。
        实际部署时需要配合涂鸦消息队列或自定义 MQTT 客户端。

        Args:
            device_ids: 要订阅的设备 ID 列表
        """
        self._logger.info("已注册 %d 个设备的实时状态订阅", len(device_ids))
        # 实际实现需要 aiomqtt 或 gmqtt 库连接 MQTT broker
        # 此处为框架预留接口
        for dev_id in device_ids:
            self._logger.debug("订阅设备状态: %s", dev_id)

    # ----------------------------------------------------------
    # ChannelAdapter 抽象方法实现
    # ----------------------------------------------------------

    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        """发送消息（IoT 适配器将消息解析为设备控制指令）"""
        try:
            text = message.content.get_primary_text() if message.content else ""
            device_id = message.get_context("device_id", "")

            if not device_id:
                return SendResult(success=False, error="缺少 device_id 上下文", error_code="MISSING_TARGET")

            # 尝试解析控制指令
            commands = self._parse_control_command(text)
            if not commands:
                return SendResult(success=False, error="无法解析控制指令", error_code="INVALID_COMMAND")

            success = await self.control_device(device_id, commands)
            if success:
                return SendResult(success=True, message_id=f"ctrl_{int(time.time())}", timestamp=time.time())
            return SendResult(success=False, error="设备控制失败", error_code="CONTROL_FAILED")

        except TuyaError as exc:
            return SendResult(success=False, error=exc.message, error_code=str(exc.code))
        except Exception as exc:
            return SendResult(success=False, error=str(exc), error_code=type(exc).__name__)

    async def _receive_impl(self, payload: Optional[Dict] = None) -> ReceiveResult:
        """接收 IoT 事件（通过 webhook 或 MQTT）"""
        if not payload:
            return ReceiveResult(success=False, error="无有效载荷")

        try:
            device_id = payload.get("devId", payload.get("device_id", ""))
            data = payload.get("data", payload)

            # 构造 UniversalMessage
            from ...universal_message import (
                MessageContent,
                MessageMetadata,
                MessageDirection,
                MessageType,
            )
            content = MessageContent(text=f"[设备状态变更] {device_id}: {json.dumps(data, ensure_ascii=False)}")
            metadata = MessageMetadata(
                message_id=f"iot_{int(time.time())}",
                channel_id="tuya_iot",
                timestamp=time.time(),
                direction=MessageDirection.INBOUND,
                message_type=MessageType.SYSTEM,
                raw_event=payload,
            )
            message = UniversalMessage(content=content, metadata=metadata)
            message.set_context("device_id", device_id)

            return ReceiveResult(success=True, messages=[message], raw_payload=payload)
        except Exception as exc:
            return ReceiveResult(success=False, error=str(exc))

    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息（涂鸦平台通过 UID 标识用户）"""
        # 涂鸦 IoT 平台不提供直接的用户信息查询 API
        # 返回缓存中的设备关联信息
        for dev in self._device_cache.values():
            if dev.uid == user_id:
                return {"uid": dev.uid, "owner_id": dev.owner_id}
        return {"uid": user_id, "message": "未找到关联设备"}

    async def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """获取通道信息"""
        return {
            "channel_id": channel_id,
            "name": "涂鸦智能 IoT",
            "endpoint": self._cfg.endpoint,
            "device_count": len(self._device_cache),
        }

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    @staticmethod
    def _parse_control_command(text: str) -> List[Dict[str, Any]]:
        """解析自然语言控制指令为设备命令

        支持的指令格式:
            - "打开" / "开启" -> {"code": "switch_1", "value": true}
            - "关闭" / "关掉" -> {"code": "switch_1", "value": false}
            - "亮度 50" -> {"code": "brightness", "value": 50}
            - "温度 25" -> {"code": "temp_set", "value": 25}
        """
        import re

        text = text.strip()
        commands: List[Dict[str, Any]] = []

        # 开关指令
        if re.search(r"(打开|开启|open|turn\s*on)", text, re.IGNORECASE):
            commands.append({"code": "switch_1", "value": True})
        elif re.search(r"(关闭|关掉|close|turn\s*off|关)", text, re.IGNORECASE):
            commands.append({"code": "switch_1", "value": False})

        # 亮度指令
        brightness_match = re.search(r"亮度\s*(\d+)", text)
        if brightness_match:
            commands.append({"code": "brightness", "value": int(brightness_match.group(1))})

        # 温度指令
        temp_match = re.search(r"温度\s*(\d+)", text)
        if temp_match:
            commands.append({"code": "temp_set", "value": int(temp_match.group(1))})

        # 色温指令
        color_temp_match = re.search(r"色温\s*(\d+)", text)
        if color_temp_match:
            commands.append({"code": "color_temp", "value": int(color_temp_match.group(1))})

        return commands

    def get_config(self) -> TuyaConfig:
        """获取当前配置"""
        return self._cfg

    def get_cached_device(self, device_id: str) -> Optional[TuyaDevice]:
        """从缓存中获取设备信息"""
        return self._device_cache.get(device_id)

    async def refresh_device_cache(self) -> int:
        """刷新设备缓存，返回设备总数"""
        _, total = await self.list_devices(page_size=100)
        return total

    async def test_connection(self) -> Dict[str, Any]:
        """测试连接"""
        try:
            token = await self._fetch_token()
            devices, total = await self.list_devices(page_size=1)
            return {
                "status": "ok",
                "token_prefix": token[:16] + "...",
                "token_expires_in": int(self._token_expires_at - time.time()),
                "total_devices": total,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def __repr__(self) -> str:
        return (
            f"TuyaIoTAdapter("
            f"channel_id={self._config.channel_id!r}, "
            f"endpoint={self._cfg.endpoint!r}, "
            f"state={self._state.name})"
        )
