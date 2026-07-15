#!/usr/bin/env python3
# =============================================================================
# UFO AGI 统一框架 - Windows EXE 打包脚本
# =============================================================================
# 使用方法:
#   python build_exe.py           - 构建EXE（目录模式，推荐）
#   python build_exe.py --onefile - 构建单文件EXE
#   python build_exe.py --clean   - 清理构建文件
# =============================================================================

import os
import sys
import shutil
import subprocess
from pathlib import Path


def info(msg):
    print(f"[INFO] {msg}")

def ok(msg):
    print(f"[OK] {msg}")

def err(msg):
    print(f"[ERROR] {msg}")


def check_pyinstaller():
    try:
        import PyInstaller
        ok(f"PyInstaller {PyInstaller.__version__} 已安装")
        return True
    except ImportError:
        return False


def install_pyinstaller():
    info("安装 PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    ok("PyInstaller 安装完成")


def clean():
    info("清理构建文件...")
    for d in ["build", "dist"]:
        if os.path.exists(d):
            shutil.rmtree(d)
            info(f"  删除 {d}/")
    for f in Path(".").glob("*.spec"):
        f.unlink()
        info(f"  删除 {f}")
    ok("清理完成")


def build(onefile=False):
    info("=" * 50)
    info("  UFO AGI 统一框架 - EXE 打包")
    info("=" * 50)

    # 检查 PyInstaller
    if not check_pyinstaller():
        install_pyinstaller()

    # 检查 main.py
    if not os.path.exists("main.py"):
        err("main.py 不存在！请确保在项目根目录运行。")
        sys.exit(1)

    # 构建 PyInstaller 命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "UFO-AGI",
        "--clean",
        "--noconfirm",
    ]

    if onefile:
        cmd.append("--onefile")
        info("模式: 单文件 EXE")
    else:
        cmd.append("--onedir")
        info("模式: 目录 EXE（推荐）")

    # 添加数据文件
    data_dirs = [
        ("web", "web"),
        ("database", "database"),
    ]
    for src, dst in data_dirs:
        if os.path.exists(src):
            cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
            info(f"  添加数据: {src}/")

    # 添加单个文件
    single_files = [
        ".env.example",
        "init_database.py",
    ]
    for f in single_files:
        if os.path.exists(f):
            cmd.extend(["--add-data", f"{f}{os.pathsep}."])
            info(f"  添加文件: {f}")

    # 隐藏导入
    hidden = [
        "uvicorn", "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
        "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
        "fastapi", "pydantic", "sqlalchemy", "sqlalchemy.dialects.sqlite",
        "aiosqlite", "aiohttp", "passlib", "bcrypt", "python_multipart",
        "email_validator", "httpx", "websockets", "yaml", "toml",
        "jinja2", "orjson",
    ]
    for h in hidden:
        cmd.extend(["--hidden-import", h])

    # 排除不需要的模块
    excludes = [
        "matplotlib", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "wx", "pytest", "sphinx", "pdb",
    ]
    for e in excludes:
        cmd.extend(["--exclude-module", e])

    # 入口文件
    cmd.append("main.py")

    # 运行
    info("开始构建...")
    info(f"命令: {' '.join(cmd[:10])}...")
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        ok("=" * 50)
        ok("  构建成功！")
        ok("=" * 50)
        if onefile:
            exe_path = Path("dist/UFO-AGI.exe")
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / 1024 / 1024
                info(f"输出: dist/UFO-AGI.exe ({size_mb:.1f} MB)")
        else:
            dist_dir = Path("dist/UFO-AGI")
            if dist_dir.exists():
                total_size = sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file())
                size_mb = total_size / 1024 / 1024
                info(f"输出: dist/UFO-AGI/ ({size_mb:.1f} MB)")
                info("运行: dist/UFO-AGI/UFO-AGI.exe")
        print()
        info("使用方法:")
        info("  1. 将 dist/UFO-AGI/ 目录复制到目标电脑")
        info("  2. 双击 UFO-AGI.exe 启动")
        info("  3. 浏览器访问 http://localhost:8000")
    else:
        err("构建失败！请检查错误信息。")
        sys.exit(1)


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--clean":
            clean()
        elif arg == "--onefile":
            build(onefile=True)
        elif arg == "--help":
            print("UFO AGI 打包脚本")
            print("  python build_exe.py           - 构建目录EXE")
            print("  python build_exe.py --onefile - 构建单文件EXE")
            print("  python build_exe.py --clean   - 清理构建文件")
        else:
            err(f"未知参数: {arg}")
    else:
        build(onefile=False)


if __name__ == "__main__":
    main()
