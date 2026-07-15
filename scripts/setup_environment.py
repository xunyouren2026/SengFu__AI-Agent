#!/usr/bin/env python3
"""
Setup Environment - 环境配置脚本

提供完整的环境配置流程，包括：
- 系统依赖检查
- Python包安装
- 配置文件生成
- 数据目录创建
- 模型下载
- 环境验证

用法：
    python scripts/setup_environment.py
    python scripts/setup_environment.py --check-only
    python scripts/setup_environment.py --install-deps
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess
import platform
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SetupConfig:
    """环境配置"""
    python_version: str = "3.10"
    install_deps: bool = True
    create_dirs: bool = True
    download_models: bool = False
    generate_config: bool = True
    check_only: bool = False


class EnvironmentSetup:
    """
    环境配置类
    
    检查和配置运行环境。
    """
    
    def __init__(self, config: SetupConfig):
        """
        初始化环境配置
        
        Args:
            config: 配置选项
        """
        self.config = config
        self.checks_passed = True
        self.warnings = []
        self.errors = []
    
    def check_python_version(self) -> Tuple[bool, str]:
        """
        检查Python版本
        
        Returns:
            (是否通过, 消息)
        """
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        required = self.config.python_version
        
        if sys.version_info >= (3, 10):
            return True, f"Python版本: {current} ✓"
        else:
            return False, f"Python版本: {current} (需要 >={required})"
    
    def check_system_info(self) -> Dict[str, str]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        return {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }
    
    def check_package(self, package: str) -> Tuple[bool, str]:
        """
        检查包是否安装
        
        Args:
            package: 包名
        
        Returns:
            (是否安装, 版本信息)
        """
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            return True, f"{package}: {version} ✓"
        except ImportError:
            return False, f"{package}: 未安装"
    
    def check_all_packages(self) -> Dict[str, Tuple[bool, str]]:
        """
        检查所有必需包
        
        Returns:
            包检查结果字典
        """
        required_packages = [
            "torch",
            "numpy",
            "transformers",
            "fastapi",
            "uvicorn",
            "pydantic",
            "aiohttp",
            "pyyaml",
        ]
        
        optional_packages = [
            "gradio",
            "tensorboard",
            "wandb",
            "deepspeed",
            "flash_attn",
        ]
        
        results = {}
        
        logger.info("检查必需包...")
        for pkg in required_packages:
            installed, msg = self.check_package(pkg)
            results[f"required:{pkg}"] = (installed, msg)
            if not installed:
                self.errors.append(f"缺少必需包: {pkg}")
                self.checks_passed = False
        
        logger.info("检查可选包...")
        for pkg in optional_packages:
            installed, msg = self.check_package(pkg)
            results[f"optional:{pkg}"] = (installed, msg)
            if not installed:
                self.warnings.append(f"缺少可选包: {pkg}")
        
        return results
    
    def check_gpu(self) -> Tuple[bool, str]:
        """
        检查GPU可用性
        
        Returns:
            (是否可用, 消息)
        """
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return True, f"GPU: {gpu_name} ({gpu_memory:.1f}GB) x {gpu_count} ✓"
            else:
                return False, "GPU: 不可用（将使用CPU）"
        except ImportError:
            return False, "GPU: PyTorch未安装"
    
    def check_disk_space(self, min_gb: float = 10.0) -> Tuple[bool, str]:
        """
        检查磁盘空间
        
        Args:
            min_gb: 最小空间（GB）
        
        Returns:
            (是否足够, 消息)
        """
        try:
            import shutil
            total, used, free = shutil.disk_usage(Path.cwd())
            free_gb = free / 1024**3
            
            if free_gb >= min_gb:
                return True, f"磁盘空间: {free_gb:.1f}GB 可用 ✓"
            else:
                return False, f"磁盘空间: {free_gb:.1f}GB 可用（需要 >={min_gb}GB）"
        except Exception as e:
            return False, f"磁盘空间: 检查失败 ({e})"
    
    def install_dependencies(self) -> bool:
        """
        安装依赖
        
        Returns:
            是否成功
        """
        logger.info("安装依赖...")
        
        requirements_file = Path("requirements.txt")
        
        if not requirements_file.exists():
            logger.error("requirements.txt 不存在")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                logger.info("依赖安装成功")
                return True
            else:
                logger.error(f"依赖安装失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"依赖安装错误: {e}")
            return False
    
    def create_directories(self) -> None:
        """创建必要的目录"""
        directories = [
            "data",
            "checkpoints",
            "outputs",
            "logs",
            "configs",
            "cache",
        ]
        
        logger.info("创建目录...")
        
        for dir_name in directories:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
                logger.info(f"创建目录: {dir_name}")
            else:
                logger.info(f"目录已存在: {dir_name}")
    
    def generate_config_files(self) -> None:
        """生成配置文件"""
        logger.info("生成配置文件...")
        
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        
        # 默认配置
        default_config = {
            "model": {
                "name": "qwen-0.5b",
                "path": None,
                "device": "auto",
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "epochs": 10,
            },
            "data": {
                "path": "./data",
                "train_file": "train.json",
                "val_file": "val.json",
            },
            "output": {
                "checkpoint_dir": "./checkpoints",
                "log_dir": "./logs",
            },
        }
        
        import json
        config_file = configs_dir / "default.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"配置文件已生成: {config_file}")
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        运行所有检查
        
        Returns:
            检查结果
        """
        logger.info("=" * 60)
        logger.info("AGI Unified Framework 环境检查")
        logger.info("=" * 60)
        
        results = {
            "system": self.check_system_info(),
            "python": self.check_python_version(),
            "packages": self.check_all_packages(),
            "gpu": self.check_gpu(),
            "disk": self.check_disk_space(),
        }
        
        # 打印结果
        logger.info("\n系统信息:")
        for key, value in results["system"].items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"\nPython: {results['python'][1]}")
        
        logger.info("\n包检查:")
        for key, (installed, msg) in results["packages"].items():
            status = "✓" if installed else "✗"
            logger.info(f"  {msg} {status}")
        
        logger.info(f"\nGPU: {results['gpu'][1]}")
        logger.info(f"磁盘: {results['disk'][1]}")
        
        return results
    
    def setup(self) -> bool:
        """
        执行完整的环境配置
        
        Returns:
            是否成功
        """
        # 运行检查
        self.run_all_checks()
        
        if self.config.check_only:
            return self.checks_passed
        
        # 安装依赖
        if self.config.install_deps and not self.checks_passed:
            if not self.install_dependencies():
                return False
        
        # 创建目录
        if self.config.create_dirs:
            self.create_directories()
        
        # 生成配置
        if self.config.generate_config:
            self.generate_config_files()
        
        # 打印摘要
        logger.info("\n" + "=" * 60)
        logger.info("环境配置完成")
        logger.info("=" * 60)
        
        if self.warnings:
            logger.info("\n警告:")
            for warning in self.warnings:
                logger.info(f"  - {warning}")
        
        if self.errors:
            logger.info("\n错误:")
            for error in self.errors:
                logger.info(f"  - {error}")
            return False
        
        logger.info("\n所有检查通过！环境已准备就绪。")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AGI环境配置脚本")
    
    parser.add_argument("--check-only", action="store_true",
                       help="仅检查环境，不进行配置")
    parser.add_argument("--install-deps", action="store_true",
                       help="安装依赖")
    parser.add_argument("--no-create-dirs", action="store_true",
                       help="不创建目录")
    parser.add_argument("--python-version", type=str, default="3.10",
                       help="要求的Python版本")
    
    args = parser.parse_args()
    
    config = SetupConfig(
        python_version=args.python_version,
        install_deps=args.install_deps,
        create_dirs=not args.no_create_dirs,
        check_only=args.check_only,
    )
    
    setup = EnvironmentSetup(config)
    success = setup.setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
