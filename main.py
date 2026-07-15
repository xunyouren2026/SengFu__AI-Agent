#!/usr/bin/env python3
# =============================================================================
# UFO AGI 统一框架 - 主入口文件
# =============================================================================

import os
import sys
import subprocess
import argparse
import logging

# 项目目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, ".venv")
VENV_PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "python")

# 检查是否在虚拟环境中运行
def check_venv():
    """检查是否应该使用虚拟环境的Python"""
    # 如果已经在虚拟环境中，直接继续
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True
    
    # 如果虚拟环境存在，且当前不是虚拟环境的Python，则重启
    if os.path.exists(VENV_PYTHON) and sys.executable != VENV_PYTHON:
        return False
    
    return True

def restart_with_venv():
    """使用虚拟环境的Python重新启动"""
    print(f"[INFO] 切换到虚拟环境: {VENV_PYTHON}")
    args = [VENV_PYTHON] + sys.argv
    subprocess.call(args)
    sys.exit(0)

# 如果需要，切换到虚拟环境
if not check_venv():
    restart_with_venv()

# 确保项目根目录在Python路径中
sys.path.insert(0, PROJECT_DIR)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("UFO")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="UFO AGI 统一框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                启动应用 (http://localhost:8000)
  python main.py --port 8080    使用8080端口
  python main.py --reload       开发模式(热重载)
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--reload", action="store_true", help="开发模式(热重载)")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--init-db", action="store_true", help="仅初始化数据库")
    return parser.parse_args()


def ensure_data_dir():
    """确保数据目录存在"""
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./uploads", exist_ok=True)


def init_database():
    """初始化数据库"""
    logger.info("初始化数据库...")
    from database.connection import DatabaseManager
    from database.models import Base
    from database.seed_data import seed_all
    
    # 重要：必须先导入所有模型类，确保它们注册到 Base.metadata
    from database.models import (
        User, UserSettings, Conversation, Message,
        Model, ModelLoadBalance,
        TrainingJob, Checkpoint, GeneratedContent,
        Workflow, WorkflowExecution,
        Agent, Alliance,
        Plugin, Channel,
        Dataset, Personality,
        AuditLog, SystemSetting,
        Reflection, Memory, Goal,
        Strategy, RoutingRule, LoadBalancer, CircuitBreaker,
        AgentLog, AgentTask, Debate,
        Marketplace, TrainingLog, HPSearch,
        Role, Permission, APIKey,
        Backup, Dashboard, Alert,
        License, HelpDoc, FAQ,
    )
    
    logger.info(f"已注册的表: {list(Base.metadata.tables.keys())[:10]}...")

    db_manager = DatabaseManager(
        database_url="sqlite:///./data/agi_framework.db",
        echo=False,
    )
    db_manager.initialize()
    db_manager.create_tables(Base)
    
    # 验证表是否创建成功
    from sqlalchemy import inspect
    inspector = inspect(db_manager.engine)
    tables = inspector.get_table_names()
    logger.info(f"数据库表已创建: {tables}")

    try:
        with db_manager.session_scope() as session:
            stats = seed_all(session)
            logger.info(f"种子数据创建完成: {stats}")
    except Exception as e:
        logger.warning(f"种子数据创建失败(可忽略): {e}")

    logger.info("数据库初始化完成")


def create_app():
    """创建FastAPI应用"""
    from api.main import create_app as _create_app
    return _create_app(
        title="UFO AGI 统一框架",
        version="1.0.0",
        description="AGI Unified Framework - 多智能体协作平台",
    )


def main():
    """主函数"""
    args = parse_args()

    print()
    print("=" * 60)
    print("  🛸 UFO AGI 统一框架 v1.0.0")
    print("=" * 60)
    print()

    ensure_data_dir()

    if args.init_db:
        init_database()
        print("\n✅ 数据库初始化完成！")
        return

    db_path = "./data/agi_framework.db"
    # 每次启动都执行数据库初始化（create_tables使用checkfirst=True，不会覆盖已有表）
    init_database()

    logger.info("创建应用...")
    app = create_app()

    import uvicorn

    logger.info(f"启动服务: http://{args.host}:{args.port}")
    print()
    print(f"  🌐 Web界面:  http://localhost:{args.port}")
    print(f"  📖 API文档:  http://localhost:{args.port}/docs")
    print()
    print("  按 Ctrl+C 停止服务")
    print("-" * 60)
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info",
        access_log=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] 服务已停止")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 程序异常: {e}")
        import traceback
        traceback.print_exc()
        print("\n按回车键退出...")
        input()
        sys.exit(1)
