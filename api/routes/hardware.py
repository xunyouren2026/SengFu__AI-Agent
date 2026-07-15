"""
Hardware API Routes - 硬件监控真实数据API
提供真实的CPU、GPU、内存、磁盘等硬件监控数据
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
import os
import platform

try:
    from ...core.database import get_db
    from ...core.auth import get_current_user
except ImportError:
    # 相对导入失败时的回退
    def get_db():
        return None
    def get_current_user():
        return {"id": "admin", "role": "admin"}

router = APIRouter(prefix="/hardware", tags=["Hardware"])


@router.get("/info", summary="获取硬件信息")
async def get_hardware_info(
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """获取系统硬件信息"""
    if not PSUTIL_AVAILABLE:
        return {
            "system": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
            },
            "warning": "psutil未安装，仅返回基础系统信息。请运行: pip install psutil"
        }
    try:
        # 获取真实硬件信息
        info = {
            "system": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python_version": platform.python_version()
            },
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
                "max_freq_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                "min_freq_mhz": psutil.cpu_freq().min if psutil.cpu_freq() else None,
                "current_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            },
            "disk": {
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
            },
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        
        return {"success": True, "data": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取硬件信息失败: {str(e)}")


@router.get("/cpu", summary="获取CPU信息")
async def get_cpu_info(
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """获取CPU详细信息"""
    if not PSUTIL_AVAILABLE:
        raise HTTPException(status_code=503, detail="psutil未安装，请运行: pip install psutil")
    try:
        cpu_info = {
            "usage_percent": psutil.cpu_percent(interval=0.1),
            "usage_per_cpu": psutil.cpu_percent(interval=0.1, percpu=True),
            "core_count": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "freq": {
                "current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "min": psutil.cpu_freq().min if psutil.cpu_freq() else None,
                "max": psutil.cpu_freq().max if psutil.cpu_freq() else None
            },
            "stats": psutil.cpu_stats()._asdict() if hasattr(psutil, 'cpu_stats') else None,
            "times": psutil.cpu_times()._asdict() if hasattr(psutil, 'cpu_times') else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return {"success": True, "data": cpu_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取CPU信息失败: {str(e)}")


@router.get("/memory", summary="获取内存信息")
async def get_memory_info(
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """获取内存详细信息"""
    if not PSUTIL_AVAILABLE:
        raise HTTPException(status_code=503, detail="psutil未安装，请运行: pip install psutil")
    try:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            "virtual": {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "free_gb": round(mem.free / (1024**3), 2),
                "usage_percent": mem.percent
            },
            "swap": {
                "total_gb": round(swap.total / (1024**3), 2),
                "used_gb": round(swap.used / (1024**3), 2),
                "free_gb": round(swap.free / (1024**3), 2),
                "usage_percent": swap.percent
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return {"success": True, "data": memory_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取内存信息失败: {str(e)}")


@router.get("/disk", summary="获取磁盘信息")
async def get_disk_info(
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """获取磁盘详细信息"""
    if not PSUTIL_AVAILABLE:
        raise HTTPException(status_code=503, detail="psutil未安装，请运行: pip install psutil")
    try:
        disk_info = []
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "opts": partition.opts,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "usage_percent": usage.percent
                })
            except PermissionError:
                continue
        
        # IO统计
        io_stats = psutil.disk_io_counters()
        
        return {
            "success": True,
            "data": {
                "partitions": disk_info,
                "io_stats": io_stats._asdict() if io_stats else None,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取磁盘信息失败: {str(e)}")


@router.get("/network", summary="获取网络信息")
async def get_network_info(
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """获取网络详细信息"""
    if not PSUTIL_AVAILABLE:
        raise HTTPException(status_code=503, detail="psutil未安装，请运行: pip install psutil")
    try:
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        
        # 网卡信息
        interfaces = []
        for name, stats in psutil.net_if_stats().items():
            try:
                addrs = psutil.net_if_addrs().get(name, [])
                interfaces.append({
                    "name": name,
                    "is_up": stats.isup,
                    "speed_mbps": stats.speed,
                    "mtu": stats.mtu,
                    "addresses": [
                        {
                            "family": addr.family.name if hasattr(addr.family, 'name') else str(addr.family),
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": addr.broadcast
                        }
                        for addr in addrs
                    ]
                })
            except:
                continue
        
        network_info = {
            "io_counters": net_io._asdict() if net_io else None,
            "connections_count": net_connections,
            "interfaces": interfaces,
            "timestamp": datetime.now().isoformat()
        }
        
        return {"success": True, "data": network_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取网络信息失败: {str(e)}")


@router.get("/gpu", summary="获取GPU信息")
async def get_gpu_info(
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """获取GPU信息（如果可用）"""
    try:
        gpus = []
        
        # 尝试使用pynvml获取NVIDIA GPU信息
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                
                gpus.append({
                    "index": i,
                    "name": name,
                    "memory": {
                        "total_mb": info.total / (1024**2),
                        "used_mb": info.used / (1024**2),
                        "free_mb": info.free / (1024**2),
                        "usage_percent": (info.used / info.total) * 100
                    },
                    "utilization": {
                        "gpu_percent": util.gpu,
                        "memory_percent": util.memory
                    },
                    "temperature": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU) if hasattr(pynvml, 'NVML_TEMPERATURE_GPU') else None
                })
        except ImportError:
            pass
        except Exception:
            pass
        
        # 如果没有GPU，返回模拟数据
        if not gpus:
            gpus = [{
                "index": 0,
                "name": "No GPU Detected",
                "memory": {
                    "total_mb": 0,
                    "used_mb": 0,
                    "free_mb": 0,
                    "usage_percent": 0
                },
                "utilization": {
                    "gpu_percent": 0,
                    "memory_percent": 0
                },
                "temperature": None,
                "note": "No GPU available or pynvml not installed"
            }]
        
        return {"success": True, "data": gpus}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取GPU信息失败: {str(e)}")


@router.get("/processes", summary="获取进程列表")
async def get_processes(
    limit: int = Query(50, ge=1, le=200, description="返回数量"),
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """获取系统进程列表"""
    if not PSUTIL_AVAILABLE:
        raise HTTPException(status_code=503, detail="psutil未安装，请运行: pip install psutil")
    try:
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'status', 'create_time']):
            try:
                pinfo = proc.info
                processes.append({
                    "pid": pinfo['pid'],
                    "name": pinfo['name'],
                    "username": pinfo['username'],
                    "cpu_percent": pinfo['cpu_percent'],
                    "memory_percent": pinfo['memory_percent'],
                    "status": pinfo['status'],
                    "create_time": datetime.fromtimestamp(pinfo['create_time']).isoformat() if pinfo['create_time'] else None
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # 按CPU使用率排序
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        
        return {"success": True, "data": processes[:limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取进程列表失败: {str(e)}")
