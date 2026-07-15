# =============================================================================
# AGI Unified Framework - Gunicorn Configuration
# =============================================================================
# 生产级 Gunicorn 配置文件
# =============================================================================

import os
import ssl
import multiprocessing

# -----------------------------------------------------------------------------
# 服务器套接字配置
# -----------------------------------------------------------------------------
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
backlog = int(os.environ.get('BACKLOG', '2048'))

# -----------------------------------------------------------------------------
# Worker 进程配置
# -----------------------------------------------------------------------------
workers = int(os.environ.get('WORKERS', str(multiprocessing.cpu_count() * 2 + 1)))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = int(os.environ.get('WORKER_CONNECTIONS', '1000'))
timeout = int(os.environ.get('TIMEOUT', '120'))
graceful_timeout = int(os.environ.get('GRACEFUL_TIMEOUT', '30'))
keep_alive = int(os.environ.get('KEEP_ALIVE', '5'))
max_requests = int(os.environ.get('MAX_REQUESTS', '1000'))
max_requests_jitter = int(os.environ.get('MAX_REQUESTS_JITTER', '50'))

# 预加载应用（减少内存占用，但禁用热重载）
preload_app = True

# -----------------------------------------------------------------------------
# 日志配置
# -----------------------------------------------------------------------------
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# -----------------------------------------------------------------------------
# 进程命名
# -----------------------------------------------------------------------------
proc_name = "agi-framework"

# -----------------------------------------------------------------------------
# 服务器机制
# -----------------------------------------------------------------------------
sendfile = None  # 让 uvicorn worker 处理

# -----------------------------------------------------------------------------
# SSL/TLS 配置（可选）
# -----------------------------------------------------------------------------
_ssl_keyfile = os.environ.get('SSL_KEYFILE')
_ssl_certfile = os.environ.get('SSL_CERTFILE')

if _ssl_keyfile and _ssl_certfile:
    keyfile = _ssl_keyfile
    certfile = _ssl_certfile
    ssl_version = ssl.PROTOCOL_TLS_SERVER
    ciphers = 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384'

# -----------------------------------------------------------------------------
# 钩子函数
# -----------------------------------------------------------------------------

def on_starting(server):
    """服务器启动前"""
    pass


def post_fork(server, worker):
    """Worker fork 后"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)


def pre_exec(server):
    """Worker exec 前（仅在使用 preload_app 时有用）"""
    server.log.info("Forked child, re-executing.")


def when_ready(server):
    """服务器就绪"""
    server.log.info("AGI Framework server is ready. Spawning workers: %s", server.num_workers)


def worker_int(worker):
    """Worker 收到 INT 信号"""
    worker.log.info("Worker received INT signal")


def worker_abort(worker):
    """Worker 异常终止"""
    worker.log.info("Worker aborted (pid: %s)", worker.pid)
