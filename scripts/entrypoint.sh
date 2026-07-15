#!/bin/bash
# =============================================================================
# AGI Unified Framework - Entrypoint Script
# =============================================================================
# Docker容器入口脚本，处理初始化、数据库迁移和服务启动
# =============================================================================

set -e

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

# =============================================================================
# 环境变量默认值
# =============================================================================
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PORT="${PORT:-8000}"
export WORKERS="${WORKERS:-4}"
export TIMEOUT="${TIMEOUT:-120}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

# =============================================================================
# 初始化
# =============================================================================
log_info "Starting AGI Unified Framework..."
log_info "Environment: ${APP_ENV:-production}"
log_info "Port: ${PORT}"
log_info "Workers: ${WORKERS}"

# 创建必要目录
log_info "Creating required directories..."
mkdir -p /app/logs /app/data /app/tmp

# =============================================================================
# 数据库迁移（如果配置了）
# =============================================================================
if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
    log_info "Running database migrations..."
    if command -v alembic &> /dev/null; then
        alembic upgrade head || log_warn "Database migration failed (non-fatal)"
    else
        log_warn "Alembic not found, skipping migrations"
    fi
fi

# =============================================================================
# 启动应用
# =============================================================================
log_info "Launching application..."

# 如果有传入命令，直接执行
if [ $# -gt 0 ]; then
    log_info "Executing: $*"
    exec "$@"
fi

# 默认启动命令：使用 gunicorn 启动 FastAPI 应用
exec gunicorn \
    "api.main:create_app()" \
    --bind "0.0.0.0:${PORT}" \
    --workers "${WORKERS}" \
    --worker-class "uvicorn.workers.UvicornWorker" \
    --timeout "${TIMEOUT}" \
    --graceful-timeout "${GRACEFUL_TIMEOUT:-30}" \
    --keep-alive "${KEEP_ALIVE:-5}" \
    --log-level "${LOG_LEVEL}" \
    --access-logfile - \
    --error-logfile -
