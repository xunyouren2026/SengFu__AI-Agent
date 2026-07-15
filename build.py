#!/usr/bin/env python3
# =============================================================================
# UFO AGI 统一框架 - 打包构建脚本
# =============================================================================
# 使用方法:
#   python build.py              - 构建可执行文件
#   python build.py --onefile    - 构建单文件版本
#   python build.py --clean      - 清理构建文件
# =============================================================================

import os
import sys
import shutil
import subprocess
from pathlib import Path


def print_info(message: str) -> None:
    """打印信息消息"""
    print(f"\033[94m[INFO]\033[0m {message}")


def print_success(message: str) -> None:
    """打印成功消息"""
    print(f"\033[92m[SUCCESS]\033[0m {message}")


def print_error(message: str) -> None:
    """打印错误消息"""
    print(f"\033[91m[ERROR]\033[0m {message}")


def check_pyinstaller() -> bool:
    """检查PyInstaller是否已安装"""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False


def install_pyinstaller() -> None:
    """安装PyInstaller"""
    print_info("安装PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    print_success("PyInstaller安装完成")


def clean_build() -> None:
    """清理构建文件"""
    print_info("清理构建文件...")
    dirs_to_remove = ["build", "dist"]
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print_info(f"已删除 {dir_name}/")
    
    # 删除spec文件
    for spec_file in Path(".").glob("*.spec"):
        spec_file.unlink()
        print_info(f"已删除 {spec_file}")
    
    print_success("清理完成")


def create_spec_file(onefile: bool = False) -> str:
    """创建PyInstaller spec文件"""
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(5000)

block_cipher = None

# 数据文件
added_files = [
    ('web', 'web'),
    ('.env.example', '.'),
    ('database', 'database'),
]

# 隐藏导入
hidden_imports = [
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'fastapi',
    'fastapi.middleware',
    'fastapi.middleware.cors',
    'fastapi.middleware.gzip',
    'pydantic',
    'pydantic.deprecated',
    'pydantic.deprecated.decorator',
    'sqlalchemy',
    'sqlalchemy.ext.asyncio',
    'sqlalchemy.dialects.sqlite',
    'alembic',
    'alembic.command',
    'alembic.config',
    'jinja2',
    'jinja2.ext',
    'aiohttp',
    'aiosqlite',
    'aioredis',
    'celery',
    'celery.loaders',
    'celery.loaders.app',
    'celery.loaders.default',
    'celery.app',
    'celery.app.base',
    'celery.app.task',
    'celery.worker',
    'celery.worker.worker',
    'kombu',
    'kombu.transport',
    'kombu.transport.redis',
    'passlib',
    'passlib.handlers',
    'passlib.handlers.pbkdf2',
    'passlib.handlers.bcrypt',
    'jwt',
    'jwt.algorithms',
    'bcrypt',
    'cryptography',
    'cryptography.fernet',
    'python_multipart',
    'python_multipart.multipart',
    'email_validator',
    'orjson',
    'ujson',
    'httpx',
    'httpx._transports',
    'httpx._transports.default',
    'websockets',
    'websockets.legacy',
    'websockets.legacy.server',
    'websockets.legacy.client',
    'asyncpg',
    'psycopg2',
    'redis',
    'redis.asyncio',
    'pillow',
    'numpy',
    'pandas',
    'requests',
    'urllib3',
    'yaml',
    'toml',
    'dotenv',
]

a = Analysis(
    ['main.py'],
    pathexx=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
        'wxPython',
        'pdb',
        'pdbpp',
        'pytest',
        'unittest',
        'doctest',
        'sphinx',
        'sphinx_rtd_theme',
        'alabaster',
        'test',
        'tests',
        '_test',
        '_tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='UFO-AGI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {'onefile': 'True' if onefile else 'False'}
)
'''
    
    spec_file = "ufo_agi.spec"
    with open(spec_file, "w") as f:
        f.write(spec_content)
    
    return spec_file


def build_executable(onefile: bool = False) -> None:
    """构建可执行文件"""
    print_info("开始构建可执行文件...")
    
    # 检查/安装PyInstaller
    if not check_pyinstaller():
        install_pyinstaller()
    
    # 创建spec文件
    spec_file = create_spec_file(onefile)
    print_info(f"创建spec文件: {spec_file}")
    
    # 运行PyInstaller
    print_info("运行PyInstaller构建...")
    cmd = [sys.executable, "-m", "PyInstaller", spec_file, "--clean"]
    
    try:
        subprocess.run(cmd, check=True)
        print_success("构建完成!")
        
        # 显示输出路径
        output_dir = Path("dist")
        if output_dir.exists():
            print_info(f"输出目录: {output_dir.absolute()}")
            for item in output_dir.iterdir():
                print_info(f"  - {item.name}")
    except subprocess.CalledProcessError as e:
        print_error(f"构建失败: {e}")
        sys.exit(1)


def main() -> None:
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="UFO AGI 打包构建脚本")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="构建单文件版本"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清理构建文件"
    )
    
    args = parser.parse_args()
    
    if args.clean:
        clean_build()
    else:
        build_executable(onefile=args.onefile)


if __name__ == "__main__":
    main()
