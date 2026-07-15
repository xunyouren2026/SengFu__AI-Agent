#!/usr/bin/env python3
"""
Serve - API服务启动脚本

提供完整的API服务启动流程，包括：
- FastAPI应用启动
- 多worker支持
- 热重载开发模式
- 生产级Gunicorn配置
- 健康检查和监控端点
- SSL/HTTPS支持

用法：
    python scripts/serve.py --port 8000
    python scripts/serve.py --host 0.0.0.0 --port 8080 --workers 4
    python scripts/serve.py --dev  # 开发模式（热重载）
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import signal
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerConfig:
    """服务器配置"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        reload: bool = False,
        ssl_keyfile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        log_level: str = "info",
        timeout: int = 120,
        max_requests: int = 1000,
        keep_alive: int = 5,
        backlog: int = 2048,
        proxy_headers: bool = True,
        forwarded_allow_ips: str = "*",
    ):
        self.host = host
        self.port = port
        self.workers = workers
        self.reload = reload
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.log_level = log_level
        self.timeout = timeout
        self.max_requests = max_requests
        self.keep_alive = keep_alive
        self.backlog = backlog
        self.proxy_headers = proxy_headers
        self.forwarded_allow_ips = forwarded_allow_ips


class APIServer:
    """
    API服务器类
    
    封装FastAPI/Uvicorn/Gunicorn服务启动逻辑。
    """
    
    def __init__(self, config: ServerConfig):
        """
        初始化API服务器
        
        Args:
            config: 服务器配置
        """
        self.config = config
        self.process = None
        self.running = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，正在关闭服务...")
        self.stop()
        sys.exit(0)
    
    def check_dependencies(self) -> bool:
        """
        检查依赖是否安装
        
        Returns:
            是否所有依赖都已安装
        """
        missing = []
        
        try:
            import uvicorn
        except ImportError:
            missing.append("uvicorn")
        
        try:
            import fastapi
        except ImportError:
            missing.append("fastapi")
        
        if missing:
            logger.error(f"缺少依赖: {', '.join(missing)}")
            logger.error("请运行: pip install " + " ".join(missing))
            return False
        
        return True
    
    def find_app_module(self) -> str:
        """
        查找FastAPI应用模块
        
        Returns:
            应用模块路径（如 "api.main:app"）
        """
        # 检查可能的API入口点
        candidates = [
            "api.main:app",
            "api.main:create_app",
            "webui.gradio_app:app",
            "main:app",
        ]
        
        for candidate in candidates:
            module_path = candidate.split(":")[0]
            file_path = module_path.replace(".", "/") + ".py"
            
            if Path(file_path).exists():
                logger.info(f"找到API入口: {candidate}")
                return candidate
        
        # 默认使用api.main
        logger.warning("未找到API入口，使用默认: api.main:app")
        return "api.main:app"
    
    def start_dev_server(self) -> None:
        """
        启动开发服务器（Uvicorn + 热重载）
        """
        import uvicorn
        
        app_module = self.find_app_module()
        
        logger.info(f"启动开发服务器: {app_module}")
        logger.info(f"地址: http://{self.config.host}:{self.config.port}")
        
        uvicorn.run(
            app_module,
            host=self.config.host,
            port=self.config.port,
            reload=True,
            log_level=self.config.log_level,
            ssl_keyfile=self.config.ssl_keyfile,
            ssl_certfile=self.config.ssl_certfile,
        )
    
    def start_prod_server(self) -> None:
        """
        启动生产服务器（Gunicorn + Uvicorn workers）
        """
        app_module = self.find_app_module()
        
        # 构建Gunicorn命令
        cmd = [
            "gunicorn",
            app_module,
            "--bind", f"{self.config.host}:{self.config.port}",
            "--workers", str(self.config.workers),
            "--worker-class", "uvicorn.workers.UvicornWorker",
            "--timeout", str(self.config.timeout),
            "--max-requests", str(self.config.max_requests),
            "--keep-alive", str(self.config.keep_alive),
            "--backlog", str(self.config.backlog),
            "--log-level", self.config.log_level,
        ]
        
        # SSL配置
        if self.config.ssl_keyfile and self.config.ssl_certfile:
            cmd.extend([
                "--keyfile", self.config.ssl_keyfile,
                "--certfile", self.config.ssl_certfile,
            ])
        
        # 代理头配置
        if self.config.proxy_headers:
            cmd.append("--forwarded-allow-ips")
            cmd.append(self.config.forwarded_allow_ips)
        
        logger.info(f"启动生产服务器: {' '.join(cmd)}")
        logger.info(f"地址: http://{self.config.host}:{self.config.port}")
        logger.info(f"Workers: {self.config.workers}")
        
        self.running = True
        self.process = subprocess.Popen(cmd)
        
        # 等待进程结束
        try:
            self.process.wait()
        except KeyboardInterrupt:
            self.stop()
    
    def start_gradio_server(self) -> None:
        """
        启动Gradio服务器
        """
        try:
            import gradio as gr
        except ImportError:
            logger.error("Gradio未安装，请运行: pip install gradio")
            return
        
        # 查找Gradio应用
        gradio_file = Path("webui/gradio_app.py")
        if not gradio_file.exists():
            logger.error(f"Gradio应用不存在: {gradio_file}")
            return
        
        logger.info(f"启动Gradio服务器")
        logger.info(f"地址: http://{self.config.host}:{self.config.port}")
        
        # 使用subprocess启动Gradio
        cmd = [
            sys.executable,
            str(gradio_file),
            "--host", self.config.host,
            "--port", str(self.config.port),
        ]
        
        if self.config.reload:
            cmd.append("--reload")
        
        self.running = True
        self.process = subprocess.Popen(cmd)
        
        try:
            self.process.wait()
        except KeyboardInterrupt:
            self.stop()
    
    def start(self, server_type: str = "auto") -> None:
        """
        启动服务器
        
        Args:
            server_type: 服务器类型 ("auto", "dev", "prod", "gradio")
        """
        if not self.check_dependencies():
            return
        
        if server_type == "auto":
            server_type = "dev" if self.config.reload or self.config.workers == 1 else "prod"
        
        logger.info(f"启动 {server_type} 服务器...")
        
        if server_type == "dev":
            self.start_dev_server()
        elif server_type == "prod":
            self.start_prod_server()
        elif server_type == "gradio":
            self.start_gradio_server()
        else:
            logger.error(f"未知的服务器类型: {server_type}")
    
    def stop(self) -> None:
        """停止服务器"""
        if self.process and self.running:
            logger.info("正在停止服务器...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.running = False
            logger.info("服务器已停止")
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        import urllib.request
        import json
        
        url = f"http://{self.config.host}:{self.config.port}/health"
        
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


def create_systemd_service(config: ServerConfig, output_path: str = "/etc/systemd/system/agi-api.service") -> str:
    """
    创建Systemd服务文件
    
    Args:
        config: 服务器配置
        output_path: 输出路径
    
    Returns:
        服务文件内容
    """
    content = f"""[Unit]
Description=AGI Unified Framework API Server
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory={Path.cwd()}
Environment="PATH={sys.prefix}/bin"
ExecStart={sys.executable} scripts/serve.py --host {config.host} --port {config.port} --workers {config.workers}
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always

[Install]
WantedBy=multi-user.target
"""
    
    logger.info(f"Systemd服务文件内容已生成，保存到: {output_path}")
    return content


def create_docker_compose(config: ServerConfig) -> str:
    """
    创建Docker Compose配置
    
    Args:
        config: 服务器配置
    
    Returns:
        Docker Compose YAML内容
    """
    content = f"""version: '3.8'

services:
  api:
    build: .
    ports:
      - "{config.port}:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    
    return content


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AGI API服务启动脚本")
    
    # 基本参数
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="监听地址")
    parser.add_argument("--port", type=int, default=8000,
                       help="监听端口")
    parser.add_argument("--workers", type=int, default=1,
                       help="Worker数量（生产模式）")
    
    # 模式参数
    parser.add_argument("--dev", action="store_true",
                       help="开发模式（热重载）")
    parser.add_argument("--prod", action="store_true",
                       help="生产模式（Gunicorn）")
    parser.add_argument("--gradio", action="store_true",
                       help="启动Gradio界面")
    
    # SSL参数
    parser.add_argument("--ssl-keyfile", type=str, default=None,
                       help="SSL私钥文件")
    parser.add_argument("--ssl-certfile", type=str, default=None,
                       help="SSL证书文件")
    
    # 日志参数
    parser.add_argument("--log-level", type=str, default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="日志级别")
    
    # 超时参数
    parser.add_argument("--timeout", type=int, default=120,
                       help="请求超时时间（秒）")
    
    # 工具命令
    parser.add_argument("--generate-systemd", action="store_true",
                       help="生成Systemd服务文件")
    parser.add_argument("--generate-docker-compose", action="store_true",
                       help="生成Docker Compose文件")
    
    args = parser.parse_args()
    
    # 创建配置
    config = ServerConfig(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.dev,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        log_level=args.log_level,
        timeout=args.timeout,
    )
    
    # 工具命令
    if args.generate_systemd:
        content = create_systemd_service(config)
        print(content)
        return
    
    if args.generate_docker_compose:
        content = create_docker_compose(config)
        print(content)
        return
    
    # 创建并启动服务器
    server = APIServer(config)
    
    # 确定服务器类型
    if args.gradio:
        server_type = "gradio"
    elif args.prod:
        server_type = "prod"
    elif args.dev:
        server_type = "dev"
    else:
        server_type = "auto"
    
    server.start(server_type)


if __name__ == "__main__":
    main()
