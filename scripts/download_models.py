#!/usr/bin/env python3
"""
Download Models - 模型下载脚本

提供完整的模型下载流程，包括：
- 从HuggingFace下载预训练模型
- 从本地镜像下载
- 模型校验
- 断点续传
- 多模型批量下载

用法：
    python scripts/download_models.py --model qwen-0.5b
    python scripts/download_models.py --models qwen-0.5b,llama-7b,mistral-7b
    python scripts/download_models.py --list
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 预定义模型列表
AVAILABLE_MODELS = {
    # Qwen系列
    "qwen-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "size_gb": 1.0,
        "description": "Qwen2.5 0.5B参数模型",
    },
    "qwen-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B",
        "size_gb": 3.0,
        "description": "Qwen2.5 1.5B参数模型",
    },
    "qwen-7b": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "size_gb": 15.0,
        "description": "Qwen2.5 7B参数模型",
    },
    # Llama系列
    "llama-7b": {
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "size_gb": 13.0,
        "description": "Llama 2 7B模型",
    },
    "llama-13b": {
        "hf_id": "meta-llama/Llama-2-13b-hf",
        "size_gb": 25.0,
        "description": "Llama 2 13B模型",
    },
    # Mistral系列
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "size_gb": 14.0,
        "description": "Mistral 7B模型",
    },
    # Phi系列
    "phi-2": {
        "hf_id": "microsoft/phi-2",
        "size_gb": 5.0,
        "description": "Microsoft Phi-2模型",
    },
    # TinyLlama
    "tinyllama": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "size_gb": 1.5,
        "description": "TinyLlama 1.1B模型",
    },
}


@dataclass
class DownloadConfig:
    """下载配置"""
    models_dir: str = "./models"
    cache_dir: str = "./cache"
    hf_token: Optional[str] = None
    mirror: Optional[str] = None  # 镜像源
    verify: bool = True
    resume: bool = True


class ModelDownloader:
    """
    模型下载器
    
    从HuggingFace或其他源下载预训练模型。
    """
    
    def __init__(self, config: DownloadConfig):
        """
        初始化下载器
        
        Args:
            config: 下载配置
        """
        self.config = config
        self.models_dir = Path(config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available_models(self) -> None:
        """列出所有可用模型"""
        print("\n可用模型列表:")
        print("=" * 70)
        print(f"{'名称':<15} {'大小(GB)':<12} {'描述'}")
        print("-" * 70)
        
        for name, info in AVAILABLE_MODELS.items():
            print(f"{name:<15} {info['size_gb']:<12.1f} {info['description']}")
        
        print("=" * 70)
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型信息字典
        """
        if model_name in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_name]
        
        # 尝试作为HuggingFace ID解析
        if "/" in model_name:
            return {
                "hf_id": model_name,
                "size_gb": "未知",
                "description": f"自定义模型: {model_name}",
            }
        
        return None
    
    def download_from_huggingface(self, model_id: str, output_dir: Path) -> bool:
        """
        从HuggingFace下载模型
        
        Args:
            model_id: HuggingFace模型ID
            output_dir: 输出目录
        
        Returns:
            是否成功
        """
        logger.info(f"从HuggingFace下载: {model_id}")
        
        try:
            from huggingface_hub import snapshot_download
            
            # 设置Token
            if self.config.hf_token:
                os.environ["HF_TOKEN"] = self.config.hf_token
            
            # 下载模型
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(output_dir),
                resume_download=self.config.resume,
            )
            
            logger.info(f"模型已下载到: {local_path}")
            return True
            
        except ImportError:
            logger.warning("huggingface_hub未安装，尝试使用transformers")
            
            try:
                from transformers import AutoModel, AutoTokenizer
                
                # 下载模型和tokenizer
                logger.info("下载模型...")
                model = AutoModel.from_pretrained(model_id)
                
                logger.info("下载tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # 保存
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                
                logger.info(f"模型已保存到: {output_dir}")
                return True
                
            except Exception as e:
                logger.error(f"下载失败: {e}")
                return False
        
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False
    
    def download_model(self, model_name: str) -> bool:
        """
        下载单个模型
        
        Args:
            model_name: 模型名称
        
        Returns:
            是否成功
        """
        model_info = self.get_model_info(model_name)
        
        if model_info is None:
            logger.error(f"未知模型: {model_name}")
            logger.info("使用 --list 查看可用模型")
            return False
        
        logger.info(f"\n下载模型: {model_name}")
        logger.info(f"描述: {model_info['description']}")
        logger.info(f"大小: {model_info['size_gb']} GB")
        
        # 输出目录
        output_dir = self.models_dir / model_name.replace("/", "_")
        
        if output_dir.exists():
            logger.warning(f"模型目录已存在: {output_dir}")
            
            # 检查是否已下载完整
            if self.config.verify and self._verify_model(output_dir):
                logger.info("模型已存在且完整，跳过下载")
                return True
        
        # 下载
        success = self.download_from_huggingface(model_info["hf_id"], output_dir)
        
        if success and self.config.verify:
            success = self._verify_model(output_dir)
        
        return success
    
    def _verify_model(self, model_dir: Path) -> bool:
        """
        校验模型完整性
        
        Args:
            model_dir: 模型目录
        
        Returns:
            是否完整
        """
        logger.info("校验模型...")
        
        required_files = ["config.json"]
        optional_files = ["pytorch_model.bin", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        
        # 检查必需文件
        for file in required_files:
            if not (model_dir / file).exists():
                logger.error(f"缺少必需文件: {file}")
                return False
        
        # 检查至少有一个模型文件
        has_model = any((model_dir / file).exists() for file in optional_files[:2])
        if not has_model:
            logger.error("缺少模型权重文件")
            return False
        
        logger.info("模型校验通过")
        return True
    
    def download_multiple(self, model_names: List[str]) -> Dict[str, bool]:
        """
        批量下载模型
        
        Args:
            model_names: 模型名称列表
        
        Returns:
            下载结果字典
        """
        results = {}
        
        for i, model_name in enumerate(model_names):
            logger.info(f"\n[{i+1}/{len(model_names)}] 下载: {model_name}")
            results[model_name] = self.download_model(model_name)
        
        # 打印摘要
        logger.info("\n" + "=" * 60)
        logger.info("下载摘要")
        logger.info("=" * 60)
        
        for model_name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {model_name}: {status}")
        
        success_count = sum(results.values())
        logger.info(f"\n成功: {success_count}/{len(model_names)}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AGI模型下载脚本")
    
    parser.add_argument("--model", type=str, default=None,
                       help="要下载的模型名称")
    parser.add_argument("--models", type=str, default=None,
                       help="要下载的模型列表（逗号分隔）")
    parser.add_argument("--list", action="store_true",
                       help="列出所有可用模型")
    parser.add_argument("--output", type=str, default="./models",
                       help="输出目录")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="HuggingFace Token")
    parser.add_argument("--no-verify", action="store_true",
                       help="不校验模型")
    
    args = parser.parse_args()
    
    # 创建配置
    config = DownloadConfig(
        models_dir=args.output,
        hf_token=args.hf_token,
        verify=not args.no_verify,
    )
    
    downloader = ModelDownloader(config)
    
    # 列出模型
    if args.list:
        downloader.list_available_models()
        return
    
    # 下载模型
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        downloader.download_multiple(model_names)
    elif args.model:
        downloader.download_model(args.model)
    else:
        # 默认下载推荐模型
        logger.info("未指定模型，下载推荐模型: qwen-0.5b")
        downloader.download_model("qwen-0.5b")


if __name__ == "__main__":
    main()
