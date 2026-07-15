"""
系统数据服务

提供真实的系统监控数据，替换Mock数据。
使用psutil获取真实的CPU、内存、磁盘、网络等数据。
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 尝试导入psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
    logger.info(f"psutil导入成功，版本: {psutil.__version__}")
except Exception as e:
    PSUTIL_AVAILABLE = False
    logger.warning(f"psutil导入失败: {e}，系统监控将返回默认值。运行: pip install psutil")


class SystemDataService:
    """系统数据服务 - 提供真实的系统监控数据"""
    
    _instance = None
    _start_time = time.time()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 预热psutil CPU测量
            if PSUTIL_AVAILABLE:
                try:
                    psutil.cpu_percent(interval=0.2)
                except:
                    pass
        return cls._instance
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        if not PSUTIL_AVAILABLE:
            return self._get_default_metrics()
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_count = psutil.cpu_count()
            
            # 内存
            memory = psutil.virtual_memory()
            
            # 磁盘
            disk = psutil.disk_usage('/')
            
            # 网络
            net_io = psutil.net_io_counters()
            
            # 负载
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                load_avg = [0.0, 0.0, 0.0]
            
            # 运行时间
            uptime = time.time() - self._start_time
            
            return {
                "cpu_usage_percent": round(cpu_percent, 2),
                "cpu_count": cpu_count,
                "memory_usage_percent": round(memory.percent, 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_usage_percent": round(disk.percent, 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "network_in_mbps": round(net_io.bytes_recv / (1024**2), 2),
                "network_out_mbps": round(net_io.bytes_sent / (1024**2), 2),
                "load_average": [round(l, 2) for l in load_avg],
                "uptime_seconds": round(uptime, 2),
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return self._get_default_metrics()
    
    def get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """获取GPU指标"""
        # 尝试使用pynvml获取NVIDIA GPU信息
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpus.append({
                    "index": i,
                    "name": pynvml.nvmlDeviceGetName(handle).decode(),
                    "usage_percent": round(util.gpu, 2),
                    "memory_used_gb": round(mem.used / (1024**3), 2),
                    "memory_total_gb": round(mem.total / (1024**3), 2),
                    "temperature_celsius": temp,
                })
            pynvml.nvmlShutdown()
            return gpus
        except Exception:
            # 无GPU或pynvml未安装
            return []
    
    def get_process_info(self) -> List[Dict[str, Any]]:
        """获取进程信息"""
        if not PSUTIL_AVAILABLE:
            return []
        
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": round(proc.info['cpu_percent'] or 0, 2),
                        "memory_percent": round(proc.info['memory_percent'] or 0, 2),
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # 按CPU使用率排序，取前10
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            return processes[:10]
        except Exception:
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        metrics = self.get_system_metrics()
        
        # 判断健康状态
        status = "healthy"
        issues = []
        
        if metrics['cpu_usage_percent'] > 90:
            status = "warning"
            issues.append("CPU使用率过高")
        
        if metrics['memory_usage_percent'] > 90:
            status = "warning"
            issues.append("内存使用率过高")
        
        if metrics['disk_usage_percent'] > 90:
            status = "critical"
            issues.append("磁盘空间不足")
        
        return {
            "status": status,
            "issues": issues,
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """返回默认指标（psutil不可用时）"""
        return {
            "cpu_usage_percent": 0.0,
            "cpu_count": 1,
            "memory_usage_percent": 0.0,
            "memory_used_gb": 0.0,
            "memory_total_gb": 1.0,
            "disk_usage_percent": 0.0,
            "disk_used_gb": 0.0,
            "disk_total_gb": 1.0,
            "network_in_mbps": 0.0,
            "network_out_mbps": 0.0,
            "load_average": [0.0, 0.0, 0.0],
            "uptime_seconds": round(time.time() - self._start_time, 2),
        }


# 全局实例
system_service = SystemDataService()
