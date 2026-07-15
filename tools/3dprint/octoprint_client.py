"""
OctoPrint Client - OctoPrint API客户端
用于控制和管理3D打印机
"""

import json
import base64
import urllib.request
import urllib.error
import urllib.parse
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import queue

# 可选依赖: websocket-client
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    websocket = None
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)


class PrinterState(Enum):
    """打印机状态枚举"""
    OPERATIONAL = "Operational"
    PRINTING = "Printing"
    PAUSED = "Paused"
    PAUSING = "Pausing"
    CANCELLING = "Cancelling"
    ERROR = "Error"
    OFFLINE = "Offline"
    UNKNOWN = "Unknown"


class ConnectionState(Enum):
    """连接状态枚举"""
    CONNECTED = "Connected"
    CONNECTING = "Connecting"
    DISCONNECTED = "Disconnected"
    ERROR = "Error"


@dataclass
class PrinterProfile:
    """打印机配置"""
    name: str
    model: str
    axes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    heated_bed: bool = False
    heated_chamber: bool = False
    extruder_count: int = 1
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrinterProfile":
        """从字典创建"""
        return cls(
            name=data.get("name", ""),
            model=data.get("model", ""),
            axes=data.get("axes", {}),
            heated_bed=data.get("heatedBed", False),
            heated_chamber=data.get("heatedChamber", False),
            extruder_count=data.get("extruder", {}).get("count", 1)
        )


@dataclass
class TemperatureData:
    """温度数据"""
    actual: float
    target: float
    offset: float = 0.0


@dataclass
class JobStatus:
    """打印任务状态"""
    state: str
    file_name: str
    file_size: int
    file_date: Optional[float]
    estimated_print_time: float
    progress: float
    print_time: float
    print_time_left: float
    print_time_left_origin: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobStatus":
        """从字典创建"""
        job = data.get("job", {})
        progress = data.get("progress", {})
        
        return cls(
            state=data.get("state", ""),
            file_name=job.get("file", {}).get("name", ""),
            file_size=job.get("file", {}).get("size", 0),
            file_date=job.get("file", {}).get("date"),
            estimated_print_time=job.get("estimatedPrintTime", 0),
            progress=progress.get("completion", 0) or 0,
            print_time=progress.get("printTime", 0) or 0,
            print_time_left=progress.get("printTimeLeft", 0) or 0,
            print_time_left_origin=progress.get("printTimeLeftOrigin", "")
        )


@dataclass
class FileEntry:
    """文件条目"""
    name: str
    path: str
    display_name: str
    size: int
    date: float
    type: str
    hash: Optional[str] = None
    estimated_print_time: Optional[float] = None
    layer_height: Optional[float] = None
    height: Optional[float] = None
    filament_used: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileEntry":
        """从字典创建"""
        return cls(
            name=data.get("name", ""),
            path=data.get("path", ""),
            display_name=data.get("display", ""),
            size=data.get("size", 0),
            date=data.get("date", 0),
            type=data.get("type", ""),
            hash=data.get("hash"),
            estimated_print_time=data.get("gcodeAnalysis", {}).get("estimatedPrintTime"),
            layer_height=data.get("gcodeAnalysis", {}).get("dimensions", {}).get("layerHeight"),
            height=data.get("gcodeAnalysis", {}).get("dimensions", {}).get("height"),
            filament_used=data.get("gcodeAnalysis", {}).get("filament", {}).get("tool0", {}).get("length")
        )


class OctoPrintClient:
    """OctoPrint API客户端"""
    
    def __init__(self, host: str, api_key: str, port: int = 80,
                 timeout: int = 30, verify_ssl: bool = True):
        self.host = host
        self.api_key = api_key
        self.port = port
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        self._base_url = f"{'https' if verify_ssl else 'http'}://{host}:{port}/api"
        self._ws_url = f"wss://{host}:{port}/sock/websocket"
        
        self._ws: Optional[Any] = None
        self._ws_connected: bool = False
        self._ws_thread: Optional[threading.Thread] = None
        self._message_queue: queue.Queue = queue.Queue()
        self._callbacks: Dict[str, List[Callable]] = {}
    
    def _make_request(self, method: str, endpoint: str,
                      data: Optional[Dict[str, Any]] = None,
                      params: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """发送API请求"""
        url = f"{self._base_url}/{endpoint}"
        
        if params:
            url += "?" + urllib.parse.urlencode(params)
        
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        request_data = json.dumps(data).encode() if data else None
        
        try:
            request = urllib.request.Request(url, data=request_data, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read()
                return json.loads(content) if content else {}
        except urllib.error.HTTPError as e:
            error_body = e.read()
            logger.error(f"OctoPrint API error: {e.code} - {error_body}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def get_version(self) -> Optional[Dict[str, Any]]:
        """获取OctoPrint版本信息"""
        return self._make_request("GET", "version")
    
    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """获取服务器信息"""
        return self._make_request("GET", "server")
    
    # ==================== 打印机操作 ====================
    
    def get_printer_state(self) -> Optional[Dict[str, Any]]:
        """获取打印机状态"""
        return self._make_request("GET", "printer")
    
    def get_printer_profiles(self) -> List[PrinterProfile]:
        """获取打印机配置列表"""
        result = self._make_request("GET", "printerprofiles")
        profiles = []
        
        if result and "profiles" in result:
            for profile_data in result["profiles"].values():
                profiles.append(PrinterProfile.from_dict(profile_data))
        
        return profiles
    
    def get_current_profile(self) -> Optional[PrinterProfile]:
        """获取当前打印机配置"""
        result = self._make_request("GET", "printerprofiles")
        
        if result and "profiles" in result:
            for profile_data in result["profiles"].values():
                if profile_data.get("current", False):
                    return PrinterProfile.from_dict(profile_data)
        
        return None
    
    def get_temperatures(self) -> Dict[str, TemperatureData]:
        """获取温度数据"""
        result = self._make_request("GET", "printer")
        temperatures = {}
        
        if result and "temperature" in result:
            temp_data = result["temperature"]
            
            if "tool0" in temp_data:
                tool = temp_data["tool0"]
                temperatures["tool0"] = TemperatureData(
                    actual=tool.get("actual", 0),
                    target=tool.get("target", 0),
                    offset=tool.get("offset", 0)
                )
            
            if "bed" in temp_data:
                bed = temp_data["bed"]
                temperatures["bed"] = TemperatureData(
                    actual=bed.get("actual", 0),
                    target=bed.get("target", 0),
                    offset=bed.get("offset", 0)
                )
            
            if "chamber" in temp_data:
                chamber = temp_data["chamber"]
                temperatures["chamber"] = TemperatureData(
                    actual=chamber.get("actual", 0),
                    target=chamber.get("target", 0),
                    offset=chamber.get("offset", 0)
                )
        
        return temperatures
    
    def set_tool_temperature(self, temperature: int, tool: int = 0) -> bool:
        """设置挤出头温度"""
        data = {"command": "target", f"targets": {f"tool{tool}": temperature}}
        result = self._make_request("POST", "printer/tool", data)
        return result is not None
    
    def set_bed_temperature(self, temperature: int) -> bool:
        """设置热床温度"""
        data = {"command": "target", "target": temperature}
        result = self._make_request("POST", "printer/bed", data)
        return result is not None
    
    def set_chamber_temperature(self, temperature: int) -> bool:
        """设置腔室温度"""
        data = {"command": "target", "target": temperature}
        result = self._make_request("POST", "printer/chamber", data)
        return result is not None
    
    def get_toolhead_position(self) -> Optional[Dict[str, float]]:
        """获取打印头位置"""
        result = self._make_request("GET", "printer/printhead")
        
        if result:
            return {
                "x": result.get("x", 0),
                "y": result.get("y", 0),
                "z": result.get("z", 0),
                "e": result.get("e", 0)
            }
        return None
    
    def jog(self, x: Optional[float] = None, y: Optional[float] = None,
            z: Optional[float] = None, speed: Optional[int] = None) -> bool:
        """手动移动打印头"""
        data = {"command": "jog"}
        
        if x is not None:
            data["x"] = x
        if y is not None:
            data["y"] = y
        if z is not None:
            data["z"] = z
        if speed is not None:
            data["speed"] = speed
        
        result = self._make_request("POST", "printer/printhead", data)
        return result is not None
    
    def home(self, axes: str = "XYZ") -> bool:
        """归零"""
        data = {"command": "home", "axes": list(axes)}
        result = self._make_request("POST", "printer/printhead", data)
        return result is not None
    
    def send_gcode(self, commands: List[str]) -> bool:
        """发送G-code命令"""
        data = {"command": "home"}  # 占位
        
        for cmd in commands:
            data = {"command": cmd}
            result = self._make_request("POST", "printer/command", data)
            if result is None:
                return False
        
        return True
    
    def extrude(self, amount: float, speed: Optional[int] = None) -> bool:
        """挤出/回抽"""
        data = {"command": "extrude", "amount": amount}
        if speed is not None:
            data["speed"] = speed
        
        result = self._make_request("POST", "printer/tool", data)
        return result is not None
    
    def select_tool(self, tool: int) -> bool:
        """选择挤出头"""
        data = {"command": "select", "tool": f"tool{tool}"}
        result = self._make_request("POST", "printer/tool", data)
        return result is not None
    
    # ==================== 连接管理 ====================
    
    def connect_printer(self, port: Optional[str] = None,
                        baudrate: Optional[int] = None,
                        printer_profile: Optional[str] = None,
                        save: bool = False) -> bool:
        """连接打印机"""
        data = {"command": "connect"}
        
        if port:
            data["port"] = port
        if baudrate:
            data["baudrate"] = baudrate
        if printer_profile:
            data["printerProfile"] = printer_profile
        if save:
            data["save"] = True
        
        result = self._make_request("POST", "connection", data)
        return result is not None
    
    def disconnect_printer(self) -> bool:
        """断开打印机连接"""
        data = {"command": "disconnect"}
        result = self._make_request("POST", "connection", data)
        return result is not None
    
    def get_connection_status(self) -> Optional[Dict[str, Any]]:
        """获取连接状态"""
        return self._make_request("GET", "connection")
    
    # ==================== 打印任务 ====================
    
    def get_job_status(self) -> Optional[JobStatus]:
        """获取打印任务状态"""
        result = self._make_request("GET", "job")
        
        if result:
            return JobStatus.from_dict(result)
        return None
    
    def start_print(self, file_path: str, select: bool = True) -> bool:
        """开始打印"""
        if select:
            # 先选择文件
            data = {"command": "select", "print": True}
            result = self._make_request("POST", f"files/{file_path}", data)
        else:
            data = {"command": "start"}
            result = self._make_request("POST", "job", data)
        
        return result is not None
    
    def pause_print(self) -> bool:
        """暂停打印"""
        data = {"command": "pause", "action": "pause"}
        result = self._make_request("POST", "job", data)
        return result is not None
    
    def resume_print(self) -> bool:
        """恢复打印"""
        data = {"command": "pause", "action": "resume"}
        result = self._make_request("POST", "job", data)
        return result is not None
    
    def cancel_print(self) -> bool:
        """取消打印"""
        data = {"command": "cancel"}
        result = self._make_request("POST", "job", data)
        return result is not None
    
    def restart_print(self) -> bool:
        """重新打印"""
        data = {"command": "restart"}
        result = self._make_request("POST", "job", data)
        return result is not None
    
    # ==================== 文件管理 ====================
    
    def list_files(self, location: str = "local") -> List[FileEntry]:
        """列出文件"""
        result = self._make_request("GET", "files")
        files = []
        
        if result and "files" in result:
            for loc_data in result["files"]:
                if loc_data.get("origin") == location or location == "all":
                    if "files" in loc_data:
                        for file_data in loc_data["files"]:
                            if file_data.get("type") == "machinecode":
                                files.append(FileEntry.from_dict(file_data))
                    elif loc_data.get("type") == "machinecode":
                        files.append(FileEntry.from_dict(loc_data))
        
        return files
    
    def get_file_info(self, location: str, path: str) -> Optional[FileEntry]:
        """获取文件信息"""
        result = self._make_request("GET", f"files/{location}/{path}")
        
        if result:
            return FileEntry.from_dict(result)
        return None
    
    def upload_file(self, file_path: str, target: str = "local",
                    select: bool = False, print_: bool = False) -> bool:
        """上传文件"""
        import os
        
        filename = os.path.basename(file_path)
        url = f"{self._base_url}/files/{target}"
        
        # 构建multipart表单数据
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(file_content)
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        }
        
        try:
            request = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return True
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return False
    
    def delete_file(self, location: str, path: str) -> bool:
        """删除文件"""
        result = self._make_request("DELETE", f"files/{location}/{path}")
        return result is not None
    
    def select_file(self, location: str, path: str, print_: bool = False) -> bool:
        """选择文件"""
        data = {"command": "select", "print": print_}
        result = self._make_request("POST", f"files/{location}/{path}", data)
        return result is not None
    
    def move_file(self, location: str, path: str, destination: str) -> bool:
        """移动文件"""
        data = {"command": "move", "destination": destination}
        result = self._make_request("POST", f"files/{location}/{path}", data)
        return result is not None
    
    def copy_file(self, location: str, path: str, destination: str) -> bool:
        """复制文件"""
        data = {"command": "copy", "destination": destination}
        result = self._make_request("POST", f"files/{location}/{path}", data)
        return result is not None
    
    # ==================== 设置 ====================
    
    def get_settings(self) -> Optional[Dict[str, Any]]:
        """获取设置"""
        return self._make_request("GET", "settings")
    
    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """更新设置"""
        result = self._make_request("POST", "settings", settings)
        return result is not None
    
    # ==================== 系统命令 ====================
    
    def get_system_commands(self) -> List[Dict[str, str]]:
        """获取系统命令列表"""
        result = self._make_request("GET", "system/commands")
        commands = []
        
        if result and "core" in result:
            for cmd in result["core"]:
                commands.append({
                    "name": cmd.get("name", ""),
                    "action": cmd.get("action", ""),
                    "source": "core"
                })
        
        return commands
    
    def execute_system_command(self, source: str, action: str) -> bool:
        """执行系统命令"""
        data = {"source": source, "action": action}
        result = self._make_request("POST", "system/commands", data)
        return result is not None
    
    # ==================== WebSocket ====================
    
    def connect_websocket(self, on_message: Optional[Callable] = None,
                          on_open: Optional[Callable] = None,
                          on_close: Optional[Callable] = None,
                          on_error: Optional[Callable] = None) -> bool:
        """连接WebSocket"""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("websocket-client 不可用，无法连接 WebSocket")
            return False
        
        if self._ws_connected:
            return True
        
        def ws_on_message(ws, message):
            try:
                data = json.loads(message)
                self._message_queue.put(data)
                
                # 触发回调
                if "current" in data:
                    self._trigger_callback("state_update", data["current"])
                if "event" in data:
                    self._trigger_callback("event", data["event"])
                
                if on_message:
                    on_message(data)
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def ws_on_open(ws):
            self._ws_connected = True
            logger.info("WebSocket connected")
            if on_open:
                on_open()
        
        def ws_on_close(ws, close_status_code, close_msg):
            self._ws_connected = False
            logger.info("WebSocket disconnected")
            if on_close:
                on_close()
        
        def ws_on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            if on_error:
                on_error(error)
        
        try:
            self._ws = websocket.WebSocketApp(
                self._ws_url,
                header={"X-Api-Key": self.api_key},
                on_open=ws_on_open,
                on_message=ws_on_message,
                on_close=ws_on_close,
                on_error=ws_on_error
            )
            
            self._ws_thread = threading.Thread(
                target=self._ws.run_forever,
                daemon=True
            )
            self._ws_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            return False
    
    def disconnect_websocket(self) -> None:
        """断开WebSocket连接"""
        if self._ws:
            self._ws.close()
            self._ws = None
            self._ws_connected = False
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """注册回调函数"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """注销回调函数"""
        if event_type in self._callbacks:
            try:
                self._callbacks[event_type].remove(callback)
            except ValueError:
                pass
    
    def _trigger_callback(self, event_type: str, data: Any) -> None:
        """触发回调"""
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def get_last_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """获取最后一条消息"""
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    # ==================== 便捷方法 ====================
    
    def is_printing(self) -> bool:
        """检查是否正在打印"""
        status = self.get_job_status()
        return status is not None and status.state == "Printing"
    
    def is_paused(self) -> bool:
        """检查是否已暂停"""
        status = self.get_job_status()
        return status is not None and status.state == "Paused"
    
    def is_operational(self) -> bool:
        """检查打印机是否就绪"""
        state = self.get_printer_state()
        if state and "state" in state:
            return state["state"].get("text") == "Operational"
        return False
    
    def get_progress(self) -> float:
        """获取打印进度"""
        status = self.get_job_status()
        return status.progress if status else 0.0
    
    def wait_for_print_complete(self, timeout: float = 3600,
                                 progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """等待打印完成"""
        start_time = time.time()
        last_progress = 0.0
        
        while time.time() - start_time < timeout:
            status = self.get_job_status()
            
            if status is None:
                time.sleep(5)
                continue
            
            if status.state == "Operational":
                return True
            
            if status.state in ("Error", "Cancelled"):
                return False
            
            if progress_callback and status.progress != last_progress:
                progress_callback(status.progress)
                last_progress = status.progress
            
            time.sleep(5)
        
        return False
