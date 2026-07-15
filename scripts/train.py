#!/usr/bin/env python3
"""
Train - 高级模型训练脚本

提供完整的模型训练流程，包括：
- 配置加载和验证
- 数据集准备
- 模型初始化（支持LoRA/QLoRA）
- DeepSpeed集成
- 训练循环（支持持续学习）
- Wandb/TensorBoard日志
- 检查点保存（支持断点续训和自动恢复）
- EWC防遗忘支持

用法：
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --model qwen-0.5b --data ./data --epochs 10
    python scripts/train.py --lora --deepspeed configs/ds_config.json
    python scripts/train.py --resume ./checkpoints/latest
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import time
import logging
import signal
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import subprocess
import warnings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class LoRAConfig:
    """LoRA配置"""
    enabled: bool = False
    r: int = 16  # LoRA秩
    alpha: int = 32  # LoRA缩放参数
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QLoRAConfig:
    """QLoRA配置"""
    enabled: bool = False
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


@dataclass
class DeepSpeedConfig:
    """DeepSpeed配置"""
    enabled: bool = False
    config_path: Optional[str] = None
    stage: int = 2  # ZeRO阶段 (0, 1, 2, 3)
    offload_optimizer: bool = False
    offload_param: bool = False


@dataclass
class LoggingConfig:
    """日志配置"""
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: str = "ufo-agi-training"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500


@dataclass
class CheckpointConfig:
    """检查点配置"""
    save_dir: str = "./checkpoints"
    save_total_limit: int = 3  # 保留的检查点数量
    resume_from_checkpoint: Optional[str] = None
    auto_resume: bool = True  # 自动从最新检查点恢复
    save_on_interrupt: bool = True  # 中断时保存检查点


@dataclass
class TrainConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "qwen-0.5b"
    model_path: Optional[str] = None
    pretrained: bool = True
    trust_remote_code: bool = True
    
    # 数据配置
    data_path: str = "./data"
    train_file: str = "train.json"
    val_file: str = "val.json"
    max_seq_length: int = 2048
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    
    # 训练配置
    epochs: int = 10
    max_steps: int = -1  # -1表示使用epochs
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"  # linear, cosine, polynomial, constant
    
    # 优化配置
    optimizer: str = "adamw_torch"  # adamw_torch, adamw_hf, adafactor
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    group_by_length: bool = True
    
    # LoRA/QLoRA配置
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    
    # DeepSpeed配置
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    
    # 日志配置
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 检查点配置
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # 持续学习配置
    continual_learning: bool = False
    ewc_lambda: float = 1e4
    tasks: List[str] = field(default_factory=list)
    
    # 输出配置
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    
    # 硬件配置
    device: str = "auto"
    local_rank: int = -1
    world_size: int = 1
    
    # 其他
    seed: int = 42
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    remove_unused_columns: bool = False


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interrupted = False
        
        # 注册信号处理
        if config.save_on_interrupt:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            atexit.register(self._cleanup)
    
    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.warning(f"收到信号 {signum}，准备保存检查点...")
        self.interrupted = True
    
    def _cleanup(self):
        """清理"""
        if self.interrupted:
            logger.info("训练中断，检查点已保存")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点路径"""
        if not self.save_dir.exists():
            return None
        
        checkpoints = [
            d for d in self.save_dir.iterdir()
            if d.is_dir() and (d / "trainer_state.json").exists()
        ]
        
        if not checkpoints:
            return None
        
        # 按修改时间排序
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def rotate_checkpoints(self):
        """轮转检查点，删除旧的"""
        if self.config.save_total_limit <= 0:
            return
        
        checkpoints = [
            d for d in self.save_dir.iterdir()
            if d.is_dir() and (d / "trainer_state.json").exists()
        ]
        
        if len(checkpoints) <= self.config.save_total_limit:
            return
        
        # 按修改时间排序，删除旧的
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        for checkpoint in checkpoints[:-self.config.save_total_limit]:
            logger.info(f"删除旧检查点: {checkpoint}")
            import shutil
            shutil.rmtree(checkpoint)


class Trainer:
    """
    高级训练器类
    
    封装完整的训练流程，支持：
    - LoRA/QLoRA微调
    - DeepSpeed分布式训练
    - Wandb/TensorBoard日志
    - 断点续训和自动恢复
    - 单任务和多任务持续学习
    """
    
    def __init__(self, config: TrainConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.ewc = None
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.training_args = None
        self.trainer = None
        
        # 检查点管理
        self.checkpoint_manager = CheckpointManager(config.checkpoint)
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 创建输出目录
        self._setup_directories()
        
        # 初始化日志
        self._setup_logging()
        
        logger.info(f"训练器初始化完成，设备: {self.device}")
        logger.info(f"配置: {json.dumps(self._config_to_dict(), indent=2, default=str)}")
    
    def _config_to_dict(self) -> Dict:
        """将配置转换为字典"""
        return {
            "model_name": self.config.model_name,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "epochs": self.config.epochs,
            "lora_enabled": self.config.lora.enabled,
            "qlora_enabled": self.config.qlora.enabled,
            "deepspeed_enabled": self.config.deepspeed.enabled,
            "fp16": self.config.fp16,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
        }
    
    def _setup_device(self) -> str:
        """设置训练设备"""
        if self.config.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return self.config.device
    
    def _set_seed(self, seed: int) -> None:
        """设置随机种子"""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                # 设置确定性行为
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    def _setup_directories(self) -> None:
        """创建输出目录"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """设置日志"""
        # Wandb
        if self.config.logging.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.logging.wandb_project,
                    entity=self.config.logging.wandb_entity,
                    tags=self.config.logging.wandb_tags,
                    config=asdict(self.config),
                )
                logger.info("Wandb日志已启用")
            except ImportError:
                logger.warning("wandb未安装，跳过Wandb日志")
        
        # TensorBoard
        if self.config.logging.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=self.config.log_dir)
                logger.info(f"TensorBoard日志已启用: {self.config.log_dir}")
            except ImportError:
                logger.warning("tensorboard未安装，跳过TensorBoard日志")
                self.tb_writer = None
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        if config_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except ImportError:
                logger.warning("PyYAML未安装，尝试解析为JSON")
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        
        logger.info(f"加载模型: {self.config.model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 量化配置（QLoRA）
        quantization_config = None
        if self.config.qlora.enabled:
            logger.info("启用QLoRA量化")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.qlora.load_in_4bit,
                bnb_4bit_compute_dtype=getattr(__import__('torch'), self.config.qlora.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.qlora.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.qlora.bnb_4bit_quant_type,
            )
        
        # 加载模型
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": getattr(__import__('torch'), "float16") if self.config.fp16 else "auto",
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # 启用梯度检查点
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        
        # 应用LoRA
        if self.config.lora.enabled or self.config.qlora.enabled:
            self._apply_lora()
        
        logger.info(f"模型加载完成，参数量: {self._get_model_params():,}")
    
    def _get_model_params(self) -> int:
        """获取模型参数量"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def _get_trainable_params(self) -> int:
        """获取可训练参数量"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _apply_lora(self):
        """应用LoRA"""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # 如果是QLoRA，先准备模型
            if self.config.qlora.enabled:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # 确定LoRA配置
            if self.config.qlora.enabled:
                lora_config = LoraConfig(
                    r=self.config.qlora.lora_r,
                    lora_alpha=self.config.qlora.lora_alpha,
                    target_modules=self.config.qlora.lora_target_modules,
                    lora_dropout=self.config.qlora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            else:
                lora_config = LoraConfig(
                    r=self.config.lora.r,
                    lora_alpha=self.config.lora.alpha,
                    target_modules=self.config.lora.target_modules,
                    lora_dropout=self.config.lora.dropout,
                    bias=self.config.lora.bias,
                    task_type=self.config.lora.task_type,
                )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("LoRA配置已应用")
        except ImportError:
            logger.error("peft未安装，无法使用LoRA/QLoRA")
            raise
    
    def prepare_data(self):
        """准备训练数据"""
        from datasets import Dataset
        
        logger.info(f"加载数据集: {self.config.data_path}")
        
        data_path = Path(self.config.data_path)
        train_path = data_path / self.config.train_file
        val_path = data_path / self.config.val_file
        
        # 加载训练数据
        if not train_path.exists():
            logger.warning(f"训练数据不存在: {train_path}，将使用示例数据")
            train_data = self._create_sample_data()
        else:
            with open(train_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
        
        # 加载验证数据
        if not val_path.exists():
            logger.warning(f"验证数据不存在: {val_path}，将使用训练数据的一部分")
            val_data = train_data[:max(1, int(len(train_data) * 0.1))] if train_data else []
        else:
            with open(val_path, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
        
        # 转换为Dataset
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _create_sample_data(self) -> List[Dict]:
        """创建示例数据"""
        return [
            {
                "instruction": "请回答以下问题",
                "input": f"示例问题 {i}",
                "output": f"示例回答 {i}"
            }
            for i in range(1000)
        ]
    
    def _format_prompt(self, example: Dict) -> str:
        """格式化提示"""
        if "instruction" in example:
            prompt = f"### 指令:\n{example['instruction']}\n\n"
            if example.get("input"):
                prompt += f"### 输入:\n{example['input']}\n\n"
            prompt += f"### 回答:\n{example['output']}"
        else:
            prompt = f"### 输入:\n{example.get('input', '')}\n\n### 回答:\n{example['output']}"
        return prompt
    
    def _tokenize_function(self, examples):
        """分词函数"""
        prompts = [self._format_prompt({"input": inp, "output": out}) 
                   for inp, out in zip(examples.get("input", examples.get("instruction", [])), 
                                       examples.get("output", []))]
        
        result = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    def setup_ewc(self) -> None:
        """设置EWC持续学习"""
        if not self.config.continual_learning:
            return
        
        logger.info("设置EWC持续学习")
        
        try:
            from core.balance_layer.ewc.ewc import EWCLoss
            self.ewc = EWCLoss(
                self.model,
                ewc_lambda=self.config.ewc_lambda
            )
        except ImportError:
            logger.warning("EWC模块未找到，持续学习功能禁用")
            self.ewc = None
    
    def _create_training_arguments(self) -> Any:
        """创建训练参数"""
        from transformers import TrainingArguments
        
        # 确定报告目标
        report_to = []
        if self.config.logging.use_wandb:
            report_to.append("wandb")
        if self.config.logging.use_tensorboard:
            report_to.append("tensorboard")
        
        # DeepSpeed配置
        deepspeed = None
        if self.config.deepspeed.enabled:
            if self.config.deepspeed.config_path:
                deepspeed = self.config.deepspeed.config_path
            else:
                # 生成DeepSpeed配置
                deepspeed = self._generate_deepspeed_config()
        
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint.save_dir,
            num_train_epochs=self.config.epochs if self.config.max_steps < 0 else None,
            max_steps=self.config.max_steps if self.config.max_steps > 0 else None,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging.log_interval,
            eval_steps=self.config.logging.eval_interval,
            save_steps=self.config.logging.save_interval,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.config.checkpoint.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            group_by_length=self.config.group_by_length,
            report_to=report_to if report_to else None,
            remove_unused_columns=self.config.remove_unused_columns,
            seed=self.config.seed,
            local_rank=self.config.local_rank,
            deepspeed=deepspeed,
            logging_dir=self.config.log_dir,
        )
        
        return training_args
    
    def _generate_deepspeed_config(self) -> str:
        """生成DeepSpeed配置文件"""
        ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": self.config.max_grad_norm,
            "zero_allow_untested_optimizer": True,
            "fp16": {
                "enabled": self.config.fp16,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": self.config.bf16,
            },
            "zero_optimization": {
                "stage": self.config.deepspeed.stage,
                "offload_optimizer": {
                    "device": "cpu" if self.config.deepspeed.offload_optimizer else "none",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu" if self.config.deepspeed.offload_param else "none",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            "wall_clock_breakdown": False,
        }
        
        config_path = Path(self.config.checkpoint.save_dir) / "deepspeed_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)
        
        logger.info(f"DeepSpeed配置已生成: {config_path}")
        return str(config_path)
    
    def train(self) -> Dict[str, Any]:
        """
        执行完整训练流程
        
        Returns:
            训练结果
        """
        logger.info("=" * 60)
        logger.info("开始训练...")
        logger.info("=" * 60)
        
        # 加载模型和分词器
        self._load_model_and_tokenizer()
        
        # 准备数据
        train_dataset, val_dataset = self.prepare_data()
        
        # 分词
        logger.info("数据分词...")
        train_dataset = train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        val_dataset = val_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
        )
        
        # 设置EWC
        self.setup_ewc()
        
        # 创建训练参数
        self.training_args = self._create_training_arguments()
        
        # 创建Trainer
        from transformers import Trainer, DataCollatorForSeq2Seq
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )
        
        # 自定义Trainer以支持EWC
        class CustomTrainer(Trainer):
            def compute_loss(self_, model, inputs, return_outputs=False, **kwargs):
                loss = super().compute_loss(model, inputs, return_outputs)
                
                # 添加EWC损失
                if self.ewc is not None:
                    ewc_loss, ewc_stats = self.ewc(model, loss if not return_outputs else loss[0])
                    if return_outputs:
                        return (ewc_loss, loss[1])
                    return ewc_loss
                
                return loss
        
        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # 检查是否需要恢复训练
        resume_from_checkpoint = None
        if self.config.checkpoint.resume_from_checkpoint:
            resume_from_checkpoint = self.config.checkpoint.resume_from_checkpoint
        elif self.config.checkpoint.auto_resume:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"自动恢复训练: {latest_checkpoint}")
                resume_from_checkpoint = latest_checkpoint
        
        # 开始训练
        logger.info("=" * 60)
        logger.info("开始训练循环")
        logger.info("=" * 60)
        
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 保存最终模型
        final_model_path = Path(self.config.output_dir) / "final_model"
        self.trainer.save_model(final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")
        
        # 保存训练指标
        metrics = train_result.metrics
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        
        # 轮转检查点
        self.checkpoint_manager.rotate_checkpoints()
        
        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info(f"训练损失: {metrics.get('train_loss', 'N/A')}")
        logger.info(f"训练步数: {metrics.get('train_runtime', 'N/A')}")
        logger.info("=" * 60)
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if self.trainer is None:
            logger.error("训练器未初始化，无法评估")
            return {}
        
        logger.info("开始评估...")
        eval_results = self.trainer.evaluate()
        logger.info(f"评估结果: {eval_results}")
        return eval_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AGI模型训练脚本（支持LoRA/QLoRA/DeepSpeed）")
    
    # 配置文件
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径（YAML或JSON）")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="qwen-0.5b",
                       help="模型名称或路径")
    parser.add_argument("--model-path", type=str, default=None,
                       help="预训练模型路径")
    
    # 数据参数
    parser.add_argument("--data", type=str, default="./data",
                       help="数据目录")
    parser.add_argument("--train-file", type=str, default="train.json",
                       help="训练数据文件")
    parser.add_argument("--val-file", type=str, default="val.json",
                       help="验证数据文件")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                       help="最大序列长度")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="梯度累积步数")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--max-steps", type=int, default=-1,
                       help="最大训练步数（覆盖epochs）")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                       help="预热步数")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="梯度裁剪阈值")
    
    # LoRA参数
    parser.add_argument("--lora", action="store_true",
                       help="启用LoRA")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA秩")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA缩放参数")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,v_proj,k_proj,o_proj",
                       help="LoRA目标模块（逗号分隔）")
    
    # QLoRA参数
    parser.add_argument("--qlora", action="store_true",
                       help="启用QLoRA（4bit量化）")
    parser.add_argument("--qlora-r", type=int, default=16,
                       help="QLoRA秩")
    parser.add_argument("--qlora-alpha", type=int, default=32,
                       help="QLoRA缩放参数")
    
    # DeepSpeed参数
    parser.add_argument("--deepspeed", action="store_true",
                       help="启用DeepSpeed")
    parser.add_argument("--deepspeed-config", type=str, default=None,
                       help="DeepSpeed配置文件路径")
    parser.add_argument("--deepspeed-stage", type=int, default=2,
                       help="DeepSpeed ZeRO阶段 (0, 1, 2, 3)")
    parser.add_argument("--offload-optimizer", action="store_true",
                       help="启用优化器状态卸载到CPU")
    parser.add_argument("--offload-param", action="store_true",
                       help="启用参数卸载到CPU")
    
    # 日志参数
    parser.add_argument("--wandb", action="store_true",
                       help="启用Wandb日志")
    parser.add_argument("--wandb-project", type=str, default="ufo-agi-training",
                       help="Wandb项目名称")
    parser.add_argument("--tensorboard", action="store_true",
                       help="启用TensorBoard日志")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="日志记录间隔")
    parser.add_argument("--eval-interval", type=int, default=100,
                       help="评估间隔")
    parser.add_argument("--save-interval", type=int, default=500,
                       help="保存间隔")
    
    # 检查点参数
    parser.add_argument("--output", type=str, default="./outputs",
                       help="输出目录")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="检查点目录")
    parser.add_argument("--save-total-limit", type=int, default=3,
                       help="保留的检查点数量")
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练的检查点路径")
    parser.add_argument("--auto-resume", action="store_true", default=True,
                       help="自动从最新检查点恢复")
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume",
                       help="禁用自动恢复")
    
    # 持续学习
    parser.add_argument("--continual", action="store_true",
                       help="启用持续学习")
    parser.add_argument("--ewc-lambda", type=float, default=1e4,
                       help="EWC正则化强度")
    
    # 硬件参数
    parser.add_argument("--device", type=str, default="auto",
                       help="训练设备")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="使用混合精度FP16")
    parser.add_argument("--bf16", action="store_true",
                       help="使用混合精度BF16")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="启用梯度检查点")
    parser.add_argument("--local-rank", type=int, default=-1,
                       help="分布式训练本地rank")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainConfig(
        model_name=args.model,
        model_path=args.model_path,
        data_path=args.data,
        train_file=args.train_file,
        val_file=args.val_file,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output,
        device=args.device,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        local_rank=args.local_rank,
        seed=args.seed,
        continual_learning=args.continual,
        ewc_lambda=args.ewc_lambda,
    )
    
    # LoRA配置
    if args.lora:
        config.lora.enabled = True
        config.lora.r = args.lora_r
        config.lora.alpha = args.lora_alpha
        config.lora.dropout = args.lora_dropout
        config.lora.target_modules = args.lora_target_modules.split(",")
    
    # QLoRA配置
    if args.qlora:
        config.qlora.enabled = True
        config.qlora.lora_r = args.qlora_r
        config.qlora.lora_alpha = args.qlora_alpha
    
    # DeepSpeed配置
    if args.deepspeed:
        config.deepspeed.enabled = True
        config.deepspeed.config_path = args.deepspeed_config
        config.deepspeed.stage = args.deepspeed_stage
        config.deepspeed.offload_optimizer = args.offload_optimizer
        config.deepspeed.offload_param = args.offload_param
    
    # 日志配置
    config.logging.use_wandb = args.wandb
    config.logging.wandb_project = args.wandb_project
    config.logging.use_tensorboard = args.tensorboard
    config.logging.log_interval = args.log_interval
    config.logging.eval_interval = args.eval_interval
    config.logging.save_interval = args.save_interval
    
    # 检查点配置
    config.checkpoint.save_dir = args.checkpoint_dir
    config.checkpoint.save_total_limit = args.save_total_limit
    config.checkpoint.resume_from_checkpoint = args.resume
    config.checkpoint.auto_resume = args.auto_resume
    
    # 从配置文件加载
    if args.config:
        trainer = Trainer(config)
        file_config = trainer.load_config(args.config)
        # 更新配置（这里简化处理，实际应该递归更新）
        for key, value in file_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 创建训练器并训练
    trainer = Trainer(config)
    
    try:
        results = trainer.train()
        
        # 输出结果
        print("\n" + "=" * 60)
        print("训练结果:")
        print("=" * 60)
        for key, value in results.items():
            print(f"  {key}: {value}")
        print("=" * 60)
        
        # 评估
        eval_results = trainer.evaluate()
        if eval_results:
            print("\n评估结果:")
            print("=" * 60)
            for key, value in eval_results.items():
                print(f"  {key}: {value}")
            print("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
