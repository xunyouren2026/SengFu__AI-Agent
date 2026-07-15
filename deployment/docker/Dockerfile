# =============================================================================
# AGI Unified Framework - Dockerfile
# =============================================================================
# 多阶段构建配置，优化镜像大小和安全性
# 构建阶段使用完整Python镜像，运行阶段使用精简镜像
# =============================================================================

# -----------------------------------------------------------------------------
# 阶段1: 基础依赖构建阶段 (Builder Stage)
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    cargo \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 安装Poetry
RUN pip install "poetry==$POETRY_VERSION"

# 设置工作目录
WORKDIR /app

# 复制依赖定义文件
COPY pyproject.toml poetry.lock* ./

# 安装Python依赖（不包括开发依赖）
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# -----------------------------------------------------------------------------
# 阶段2: 运行阶段 (Runtime Stage)
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    APP_USER=agi \
    APP_GROUP=agi \
    PORT=8000 \
    WORKERS=4 \
    TIMEOUT=120 \
    GRACEFUL_TIMEOUT=30 \
    KEEP_ALIVE=5 \
    LOG_LEVEL=info

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建非root用户（安全最佳实践）
RUN groupadd --gid 1000 $APP_GROUP \
    && useradd --uid 1000 --gid $APP_GROUP --shell /bin/bash --create-home $APP_USER

# 设置工作目录
WORKDIR $APP_HOME

# 从builder阶段复制Python虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 确保虚拟环境在PATH中
ENV PATH="/app/.venv/bin:$PATH"

# 复制应用代码
COPY --chown=$APP_USER:$APP_GROUP ./api ./api
COPY --chown=$APP_USER:$APP_GROUP ./core ./core
COPY --chown=$APP_USER:$APP_GROUP ./scripts ./scripts
COPY --chown=$APP_USER:$APP_GROUP ./config ./config

# 复制启动脚本
COPY --chown=$APP_USER:$APP_GROUP ./scripts/entrypoint.sh ./scripts/healthcheck.sh ./
RUN chmod +x entrypoint.sh healthcheck.sh

# 创建必要的目录并设置权限
RUN mkdir -p /app/logs /app/data /app/tmp \
    && chown -R $APP_USER:$APP_GROUP /app

# 切换到非root用户
USER $APP_USER

# 暴露应用端口
EXPOSE $PORT

# 健康检查配置
# 每30秒检查一次，超时10秒，连续3次失败视为不健康
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ./healthcheck.sh || exit 1

# 使用entrypoint脚本启动应用
ENTRYPOINT ["./entrypoint.sh"]

# 默认启动命令
CMD ["gunicorn", "-c", "config/gunicorn.conf.py", "api.main:create_app()"]

# -----------------------------------------------------------------------------
# 阶段3: 开发阶段 (Development Stage) - 可选
# -----------------------------------------------------------------------------
FROM builder AS development

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    PORT=8000

WORKDIR $APP_HOME

# 安装开发依赖
RUN poetry install --with=dev --no-root

# 复制完整代码
COPY . .

# 暴露端口
EXPOSE $PORT

# 开发模式使用uvicorn热重载
CMD ["uvicorn", "api.main:create_app()", "--host", "0.0.0.0", "--port", "8000", "--reload", "--factory"]
