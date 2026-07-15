# =============================================================================
# UFO AGI 统一框架 - Docker 镜像
# =============================================================================
# 构建命令: docker build -t ufo-agi:latest .
# 运行命令: docker run -p 8000:8000 ufo-agi:latest
# =============================================================================

FROM python:3.11-slim as builder

# 设置工作目录
WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# 生产镜像
# =============================================================================
FROM python:3.11-slim

# 设置元数据
LABEL maintainer="UFO Team"
LABEL description="UFO AGI Unified Framework"
LABEL version="1.0.0"

# 设置工作目录
WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 创建非root用户
RUN groupadd -r ufo && useradd -r -g ufo ufo

# 创建必要的目录
RUN mkdir -p /app/data /app/logs /app/uploads && \
    chown -R ufo:ufo /app

# 复制应用代码
COPY --chown=ufo:ufo . .

# 初始化数据库
RUN python init_database.py

# 切换到非root用户
USER ufo

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "main.py"]
