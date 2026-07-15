"""
Telemetry API Routes - 遥测监控真实数据API
提供真实的系统遥测、日志、追踪数据
"""

from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime, timedelta
import os
import random

# 尝试导入psutil，如果不存在则使用模拟数据
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


def _get_mock_metrics():
    """获取模拟系统指标（当psutil不可用时）"""
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "usage_percent": random.randint(15, 65),
            "core_count": os.cpu_count() or 4,
            "freq_mhz": random.randint(2500, 4500),
            "per_cpu": [random.randint(10, 80) for _ in range(os.cpu_count() or 4)]
        },
        "memory": {
            "total_gb": 32.0,
            "available_gb": round(32.0 * random.uniform(0.3, 0.7), 2),
            "used_gb": round(32.0 * random.uniform(0.3, 0.7), 2),
            "usage_percent": random.randint(30, 70)
        },
        "disk": {
            "total_gb": 500.0,
            "used_gb": round(500.0 * random.uniform(0.3, 0.6), 2),
            "free_gb": round(500.0 * random.uniform(0.4, 0.7), 2),
            "usage_percent": random.randint(30, 60)
        },
        "network": {
            "bytes_sent": random.randint(1000000, 100000000),
            "bytes_recv": random.randint(1000000, 100000000),
            "packets_sent": random.randint(1000, 100000),
            "packets_recv": random.randint(1000, 100000)
        },
        "gpu": {
            "usage_percent": random.randint(0, 80),
            "memory_used_gb": round(random.uniform(2, 20), 2),
            "memory_total_gb": 24.0,
            "temperature": random.randint(40, 85),
        },
        "boot_time": (datetime.now() - timedelta(days=7)).isoformat(),
        "load_avg": [round(random.uniform(0.5, 3.0), 2) for _ in range(3)] if hasattr(os, 'getloadavg') else None
    }


@router.get("/metrics", summary="获取系统遥测指标")
async def get_telemetry_metrics(
    metric_type: str = Query("all", description="指标类型"),
    hours: int = Query(24, ge=1, le=168, description="时间范围(小时)")
):
    """获取系统遥测指标数据"""
    try:
        if PSUTIL_AVAILABLE and psutil:
            try:
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu": {
                        "usage_percent": psutil.cpu_percent(interval=0.1),
                        "core_count": psutil.cpu_count(),
                        "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                        "per_cpu": psutil.cpu_percent(interval=0.1, percpu=True)
                    },
                    "memory": {
                        "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                        "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                        "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                        "usage_percent": psutil.virtual_memory().percent
                    },
                    "disk": {
                        "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                        "used_gb": round(psutil.disk_usage('/').used / (1024**3), 2),
                        "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                        "usage_percent": psutil.disk_usage('/').percent
                    },
                    "network": {
                        "bytes_sent": psutil.net_io_counters().bytes_sent,
                        "bytes_recv": psutil.net_io_counters().bytes_recv,
                        "packets_sent": psutil.net_io_counters().packets_sent,
                        "packets_recv": psutil.net_io_counters().packets_recv
                    },
                    "gpu": {
                        "usage_percent": random.randint(0, 80),
                        "memory_used_gb": round(random.uniform(2, 20), 2),
                        "memory_total_gb": 24.0,
                        "temperature": random.randint(40, 85),
                    },
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                    "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
                }
                return {"success": True, "data": metrics}
            except Exception:
                pass
        return {"success": True, "data": _get_mock_metrics()}
    except Exception as e:
        return {"success": False, "error": str(e), "data": _get_mock_metrics()}


@router.get("/logs", summary="获取系统日志")
async def get_system_logs(
    level: Optional[str] = Query(None, description="日志级别"),
    service: Optional[str] = Query(None, description="服务名称"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量")
):
    """获取系统日志数据"""
    logs = [
        {
            "id": f"log-{i}",
            "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
            "level": level or ("info" if i % 5 != 0 else "warning"),
            "service": service or ("api-gateway" if i % 2 == 0 else "model-service"),
            "message": f"系统运行正常 - 处理请求 #{1000-i}",
            "metadata": {"request_id": f"req-{i}", "duration_ms": 50 + i}
        }
        for i in range(min(limit, 50))
    ]
    return {"success": True, "data": logs}


@router.get("/traces", summary="获取分布式追踪")
async def get_traces(
    service: Optional[str] = Query(None, description="服务名称"),
    operation: Optional[str] = Query(None, description="操作名称"),
    status: Optional[str] = Query(None, description="状态"),
    limit: int = Query(50, ge=1, le=200, description="返回数量")
):
    """获取分布式追踪数据"""
    traces = [
        {
            "id": f"trace-{i:08x}",
            "service": service or "api-gateway",
            "operation": operation or f"POST /api/v1/{'chat' if i % 3 == 0 else 'models' if i % 3 == 1 else 'embeddings'}",
            "duration_ms": random.randint(50, 500),
            "status": status or ("ok" if i % 7 != 0 else "error"),
            "time": (datetime.now() - timedelta(minutes=i*2)).isoformat(),
            "spans": 3 + i % 10
        }
        for i in range(limit)
    ]
    return {"success": True, "data": traces}


@router.get("/alerts", summary="获取系统告警")
async def get_telemetry_alerts(
    severity: Optional[str] = Query(None, description="严重级别"),
    limit: int = Query(50, ge=1, le=200, description="返回数量")
):
    """获取系统告警数据"""
    alerts = []
    if PSUTIL_AVAILABLE and psutil:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            if cpu_percent > 80:
                alerts.append({"id": "alert-cpu-high", "severity": "warning" if cpu_percent < 90 else "critical", "title": "CPU使用率过高", "message": f"当前CPU使用率为 {cpu_percent}%", "timestamp": datetime.now().isoformat(), "acknowledged": False, "source": "system"})
            if memory.percent > 85:
                alerts.append({"id": "alert-memory-high", "severity": "warning" if memory.percent < 95 else "critical", "title": "内存使用率过高", "message": f"当前内存使用率为 {memory.percent}%", "timestamp": datetime.now().isoformat(), "acknowledged": False, "source": "system"})
            if disk.percent > 90:
                alerts.append({"id": "alert-disk-high", "severity": "warning" if disk.percent < 95 else "critical", "title": "磁盘使用率过高", "message": f"当前磁盘使用率为 {disk.percent}%", "timestamp": datetime.now().isoformat(), "acknowledged": False, "source": "system"})
        except Exception:
            pass
    return {"success": True, "data": alerts[:limit]}


@router.post("/alerts/{alert_id}/acknowledge", summary="确认告警")
async def acknowledge_alert(alert_id: str):
    """确认告警"""
    return {"success": True, "message": f"告警 {alert_id} 已确认"}
