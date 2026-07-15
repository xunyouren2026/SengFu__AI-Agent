"""
ModelConfigs - 模型配置模板模块

该模块提供了完整的深度学习模型配置管理功能，包括模型架构、训练超参数、
优化器和学习率调度配置。支持配置验证、序列化和反序列化操作。

模块路径: config/templates/model_configs.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ModelArchitecture(Enum):
    """模型架构类型枚举"""
    TRANSFORMER = "transformer"
    BERT = "bert"
    GPT = "gpt"
    T5 = "t5"
    LLAMA = "llama"
    MISTRAL = "mistral"
    RESNET = "resnet"
    VGG = "vgg"
    VISION_TRANSFORMER = "vision_transformer"
    LSTM = "lstm"
    GRU = "gru"
    UNET = "unet"
    DIFFUSION = "diffusion"
    GAN = "gan"
    VAE = "vae"


class ActivationFunction(Enum):
    """激活函数枚举"""
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"


class NormalizationType(Enum):
    """归一化类型枚举"""
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"
    RMS_NORM = "rms_norm"


class OptimizerType(Enum):
    """优化器类型枚举"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"
    LAMB = "lamb"
    LION = "lion"


class LRSchedulerType(Enum):
    """学习率调度器类型枚举"""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    STEP = "step"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    WARMUP_COSINE = "warmup_cosine"


class PrecisionType(Enum):
    """精度类型枚举"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    MIXED = "mixed"


@dataclass
class TransformerConfig:
    """Transformer模型配置类 - 配置Transformer架构模型的参数"""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    activation_function: Union[ActivationFunction, str] = ActivationFunction.GELU
    normalization_type: Union[NormalizationType, str] = NormalizationType.LAYER_NORM
    layer_norm_eps: float = 1e-5
    use_bias: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = False
    sliding_window: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    
    def __post_init__(self):
        if isinstance(self.activation_function, str):
            try:
                self.activation_function = ActivationFunction(self.activation_function)
            except ValueError:
                pass
        if isinstance(self.normalization_type, str):
            try:
                self.normalization_type = NormalizationType(self.normalization_type)
            except ValueError:
                pass
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
    
    def validate(self) -> bool:
        """验证Transformer配置的有效性"""
        if self.vocab_size < 1:
            raise ValueError(f"词表大小必须大于0")
        if self.hidden_size < 1:
            raise ValueError(f"隐藏层维度必须大于0")
        if self.num_layers < 1:
            raise ValueError(f"层数必须大于0")
        if self.num_heads < 1:
            raise ValueError(f"注意力头数必须大于0")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"隐藏层维度必须能被头数整除")
        if self.intermediate_size < 1:
            raise ValueError(f"FFN中间层维度必须大于0")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"Dropout比率必须在[0, 1]范围内")
        if not 0 <= self.attention_dropout_rate <= 1:
            raise ValueError(f"注意力Dropout比率必须在[0, 1]范围内")
        if self.num_key_value_heads > self.num_heads:
            raise ValueError(f"KV头数不能大于总头数")
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(f"头数必须能被KV头数整除")
        return True
    
    def get_head_dim(self) -> int:
        """计算每个注意力头的维度"""
        return self.hidden_size // self.num_heads
    
    def estimate_parameters(self) -> int:
        """估算模型参数量"""
        embedding_params = self.vocab_size * self.hidden_size
        position_params = self.max_position_embeddings * self.hidden_size
        attn_params = 4 * self.hidden_size * self.hidden_size
        ffn_params = self.hidden_size * self.intermediate_size * 2
        norm_params = 4 * self.hidden_size
        layer_params = attn_params + ffn_params + norm_params
        total_layer_params = self.num_layers * layer_params
        output_params = 0 if self.tie_word_embeddings else self.hidden_size * self.vocab_size
        return embedding_params + position_params + total_layer_params + output_params
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(self.activation_function, ActivationFunction):
            result['activation_function'] = self.activation_function.value
        if isinstance(self.normalization_type, NormalizationType):
            result['normalization_type'] = self.normalization_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TransformerConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> TransformerConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> TransformerConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class CNNConfig:
    """CNN模型配置类 - 配置卷积神经网络模型的参数"""
    input_channels: int = 3
    num_classes: int = 1000
    num_layers: int = 50
    base_channels: int = 64
    channel_multiplier: float = 2.0
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [2, 1, 1, 1])
    use_batch_norm: bool = True
    use_residual: bool = True
    dropout_rate: float = 0.0
    global_pooling: str = "avg"
    
    def validate(self) -> bool:
        """验证CNN配置的有效性"""
        if self.input_channels < 1:
            raise ValueError(f"输入通道数必须大于0")
        if self.num_classes < 1:
            raise ValueError(f"类别数必须大于0")
        if self.num_layers < 1:
            raise ValueError(f"层数必须大于0")
        if self.base_channels < 1:
            raise ValueError(f"基础通道数必须大于0")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"Dropout比率必须在[0, 1]范围内")
        if self.global_pooling not in ["avg", "max"]:
            raise ValueError(f"全局池化类型必须是'avg'或'max'")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CNNConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> CNNConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> CNNConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class RNNConfig:
    """RNN模型配置类 - 配置循环神经网络模型的参数"""
    vocab_size: int = 10000
    embedding_dim: int = 300
    hidden_size: int = 512
    num_layers: int = 2
    rnn_type: str = "lstm"
    bidirectional: bool = False
    dropout_rate: float = 0.3
    num_classes: int = 2
    max_seq_length: int = 512
    
    def validate(self) -> bool:
        """验证RNN配置的有效性"""
        if self.vocab_size < 1:
            raise ValueError(f"词表大小必须大于0")
        if self.hidden_size < 1:
            raise ValueError(f"隐藏层维度必须大于0")
        if self.num_layers < 1:
            raise ValueError(f"层数必须大于0")
        if self.rnn_type not in ["lstm", "gru", "rnn"]:
            raise ValueError(f"RNN类型必须是'lstm'、'gru'或'rnn'")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"Dropout比率必须在[0, 1]范围内")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RNNConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> RNNConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> RNNConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class OptimizerConfig:
    """优化器配置类 - 配置模型训练优化器的参数"""
    optimizer_type: Union[OptimizerType, str] = OptimizerType.ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    momentum: float = 0.9
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_8bit_adam: bool = False
    use_fused_optimizer: bool = True
    
    def __post_init__(self):
        if isinstance(self.optimizer_type, str):
            try:
                self.optimizer_type = OptimizerType(self.optimizer_type)
            except ValueError:
                pass
    
    def validate(self) -> bool:
        """验证优化器配置的有效性"""
        if self.learning_rate <= 0:
            raise ValueError(f"学习率必须大于0")
        if self.weight_decay < 0:
            raise ValueError(f"权重衰减不能为负数")
        if not 0 <= self.beta1 < 1:
            raise ValueError(f"beta1必须在[0, 1)范围内")
        if not 0 <= self.beta2 < 1:
            raise ValueError(f"beta2必须在[0, 1)范围内")
        if self.eps <= 0:
            raise ValueError(f"epsilon必须大于0")
        if not 0 <= self.momentum < 1:
            raise ValueError(f"动量必须在[0, 1)范围内")
        if self.max_grad_norm <= 0:
            raise ValueError(f"梯度裁剪阈值必须大于0")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"梯度累积步数必须大于等于1")
        return True
    
    def effective_batch_size(self, per_device_batch_size: int, num_devices: int = 1) -> int:
        """计算有效批次大小"""
        return per_device_batch_size * num_devices * self.gradient_accumulation_steps
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(self.optimizer_type, OptimizerType):
            result['optimizer_type'] = self.optimizer_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OptimizerConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> OptimizerConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> OptimizerConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class LRSchedulerConfig:
    """学习率调度器配置类 - 配置学习率调度策略的参数"""
    scheduler_type: Union[LRSchedulerType, str] = LRSchedulerType.COSINE
    num_warmup_steps: int = 1000
    num_training_steps: int = 100000
    num_cycles: float = 1.0
    min_lr_ratio: float = 0.1
    power: float = 1.0
    step_size: int = 1000
    gamma: float = 0.1
    patience: int = 10
    warmup_ratio: Optional[float] = None
    last_epoch: int = -1
    
    def __post_init__(self):
        if isinstance(self.scheduler_type, str):
            try:
                self.scheduler_type = LRSchedulerType(self.scheduler_type)
            except ValueError:
                pass
        if self.warmup_ratio is not None:
            self.num_warmup_steps = int(self.num_training_steps * self.warmup_ratio)
    
    def validate(self) -> bool:
        """验证学习率调度器配置的有效性"""
        if self.num_warmup_steps < 0:
            raise ValueError(f"预热步数不能为负数")
        if self.num_training_steps < 1:
            raise ValueError(f"训练步数必须大于0")
        if self.num_warmup_steps >= self.num_training_steps:
            raise ValueError(f"预热步数不能大于等于训练步数")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError(f"最小学习率比例必须在(0, 1]范围内")
        if self.power <= 0:
            raise ValueError(f"幂次必须大于0")
        if self.step_size < 1:
            raise ValueError(f"步长必须大于0")
        if not 0 < self.gamma < 1:
            raise ValueError(f"衰减系数必须在(0, 1)范围内")
        return True
    
    def get_lr_at_step(self, initial_lr: float, current_step: int) -> float:
        """计算指定步数的学习率"""
        if current_step < self.num_warmup_steps:
            return initial_lr * current_step / max(1, self.num_warmup_steps)
        
        scheduler_type = self.scheduler_type
        if isinstance(scheduler_type, str):
            scheduler_type = LRSchedulerType(scheduler_type)
        
        if scheduler_type == LRSchedulerType.CONSTANT:
            return initial_lr
        elif scheduler_type == LRSchedulerType.LINEAR:
            progress = (current_step - self.num_warmup_steps) / max(
                1, self.num_training_steps - self.num_warmup_steps
            )
            return initial_lr * (1 - progress * (1 - self.min_lr_ratio))
        elif scheduler_type == LRSchedulerType.COSINE:
            progress = (current_step - self.num_warmup_steps) / max(
                1, self.num_training_steps - self.num_warmup_steps
            )
            return initial_lr * (
                self.min_lr_ratio + (1 - self.min_lr_ratio) * 
                (1 + math.cos(math.pi * progress)) / 2
            )
        elif scheduler_type == LRSchedulerType.POLYNOMIAL:
            progress = (current_step - self.num_warmup_steps) / max(
                1, self.num_training_steps - self.num_warmup_steps
            )
            return initial_lr * (1 - progress) ** self.power
        else:
            return initial_lr
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(self.scheduler_type, LRSchedulerType):
            result['scheduler_type'] = self.scheduler_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LRSchedulerConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> LRSchedulerConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> LRSchedulerConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class TrainingConfig:
    """训练配置类 - 配置模型训练过程的参数"""
    num_epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    max_seq_length: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    dataloader_drop_last: bool = False
    dataloader_shuffle: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    resume_from_checkpoint: Optional[str] = None
    
    def validate(self) -> bool:
        """验证训练配置的有效性"""
        if self.num_epochs < 1:
            raise ValueError(f"训练轮数必须大于0")
        if self.batch_size < 1:
            raise ValueError(f"批次大小必须大于0")
        if self.eval_batch_size < 1:
            raise ValueError(f"评估批次大小必须大于0")
        if self.max_seq_length < 1:
            raise ValueError(f"最大序列长度必须大于0")
        if self.num_workers < 0:
            raise ValueError(f"工作线程数不能为负数")
        if self.prefetch_factor < 1:
            raise ValueError(f"预取因子必须大于0")
        if self.fp16 and self.bf16:
            raise ValueError("不能同时启用FP16和BF16")
        if self.save_steps < 1:
            raise ValueError(f"保存步数必须大于0")
        if self.eval_steps < 1:
            raise ValueError(f"评估步数必须大于0")
        if self.logging_steps < 1:
            raise ValueError(f"日志步数必须大于0")
        return True
    
    def get_precision_type(self) -> PrecisionType:
        """获取精度类型"""
        if self.fp16:
            return PrecisionType.FP16
        elif self.bf16:
            return PrecisionType.BF16
        else:
            return PrecisionType.FP32
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> TrainingConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> TrainingConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class ModelConfig:
    """综合模型配置类 - 整合模型架构、训练、优化器和学习率调度的统一配置类"""
    model_architecture: Union[ModelArchitecture, str] = ModelArchitecture.TRANSFORMER
    transformer: Optional[TransformerConfig] = None
    cnn: Optional[CNNConfig] = None
    rnn: Optional[RNNConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    model_name: Optional[str] = None
    model_version: str = "1.0.0"
    description: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.model_architecture, str):
            try:
                self.model_architecture = ModelArchitecture(self.model_architecture)
            except ValueError:
                pass
        arch = self.model_architecture
        if isinstance(arch, str):
            try:
                arch = ModelArchitecture(arch)
            except ValueError:
                arch = None
        if arch in [ModelArchitecture.TRANSFORMER, ModelArchitecture.GPT, 
                    ModelArchitecture.BERT, ModelArchitecture.T5,
                    ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL]:
            if self.transformer is None:
                self.transformer = TransformerConfig()
        elif arch == ModelArchitecture.RESNET or arch == ModelArchitecture.VGG:
            if self.cnn is None:
                self.cnn = CNNConfig()
        elif arch in [ModelArchitecture.LSTM, ModelArchitecture.GRU]:
            if self.rnn is None:
                self.rnn = RNNConfig()
    
    def validate(self) -> bool:
        """验证所有配置的有效性"""
        self.training.validate()
        self.optimizer.validate()
        self.lr_scheduler.validate()
        
        arch = self.model_architecture
        if isinstance(arch, str):
            try:
                arch = ModelArchitecture(arch)
            except ValueError:
                raise ValueError(f"不支持的模型架构: {self.model_architecture}")
        
        if arch in [ModelArchitecture.TRANSFORMER, ModelArchitecture.GPT, 
                    ModelArchitecture.BERT, ModelArchitecture.T5,
                    ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL]:
            if self.transformer is None:
                raise ValueError(f"{arch.value}架构需要提供transformer配置")
            self.transformer.validate()
        
        if arch == ModelArchitecture.RESNET or arch == ModelArchitecture.VGG:
            if self.cnn is None:
                raise ValueError(f"{arch.value}架构需要提供cnn配置")
            self.cnn.validate()
        
        if arch in [ModelArchitecture.LSTM, ModelArchitecture.GRU]:
            if self.rnn is None:
                raise ValueError(f"{arch.value}架构需要提供rnn配置")
            self.rnn.validate()
        
        return True
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """估算模型内存使用量(GB)"""
        arch = self.model_architecture
        if isinstance(arch, str):
            try:
                arch = ModelArchitecture(arch)
            except ValueError:
                return {"model_params_gb": 0.0, "activation_gb": 0.0, "total_gb": 0.0}
        
        if arch in [ModelArchitecture.TRANSFORMER, ModelArchitecture.GPT, 
                    ModelArchitecture.BERT, ModelArchitecture.T5] and self.transformer:
            params = self.transformer.estimate_parameters()
        else:
            params = 0
        
        bytes_per_param = 4 if not (self.training.fp16 or self.training.bf16) else 2
        model_params_gb = params * bytes_per_param / (1024**3)
        optimizer_multiplier = 2 if self.optimizer.optimizer_type in [
            OptimizerType.ADAM, OptimizerType.ADAMW
        ] else 1
        optimizer_gb = model_params_gb * optimizer_multiplier
        activation_gb = model_params_gb * 0.5
        total_gb = model_params_gb + optimizer_gb + activation_gb
        
        return {
            "model_params_gb": round(model_params_gb, 2),
            "optimizer_state_gb": round(optimizer_gb, 2),
            "activation_gb": round(activation_gb, 2),
            "total_gb": round(total_gb, 2)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model_architecture": self.model_architecture.value if isinstance(
                self.model_architecture, ModelArchitecture
            ) else self.model_architecture,
            "training": self.training.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "lr_scheduler": self.lr_scheduler.to_dict(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "description": self.description
        }
        if self.transformer:
            result["transformer"] = self.transformer.to_dict()
        if self.cnn:
            result["cnn"] = self.cnn.to_dict()
        if self.rnn:
            result["rnn"] = self.rnn.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        return cls(
            model_architecture=data.get("model_architecture", ModelArchitecture.TRANSFORMER),
            transformer=TransformerConfig.from_dict(data["transformer"]) if "transformer" in data else None,
            cnn=CNNConfig.from_dict(data["cnn"]) if "cnn" in data else None,
            rnn=RNNConfig.from_dict(data["rnn"]) if "rnn" in data else None,
            training=TrainingConfig.from_dict(data.get("training", {})),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer", {})),
            lr_scheduler=LRSchedulerConfig.from_dict(data.get("lr_scheduler", {})),
            model_name=data.get("model_name"),
            model_version=data.get("model_version", "1.0.0"),
            description=data.get("description")
        )
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> ModelConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> ModelConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


class ModelPresets:
    """预定义的模型配置模板"""
    
    @staticmethod
    def gpt2_small() -> ModelConfig:
        """GPT-2 Small配置 (124M参数)"""
        return ModelConfig(
            model_architecture=ModelArchitecture.GPT,
            transformer=TransformerConfig(
                vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12,
                intermediate_size=3072, max_position_embeddings=1024
            ),
            training=TrainingConfig(batch_size=32, max_seq_length=1024),
            optimizer=OptimizerConfig(learning_rate=5e-4),
            model_name="gpt2-small"
        )
    
    @staticmethod
    def gpt2_medium() -> ModelConfig:
        """GPT-2 Medium配置 (355M参数)"""
        return ModelConfig(
            model_architecture=ModelArchitecture.GPT,
            transformer=TransformerConfig(
                vocab_size=50257, hidden_size=1024, num_layers=24, num_heads=16,
                intermediate_size=4096, max_position_embeddings=1024
            ),
            training=TrainingConfig(batch_size=16, max_seq_length=1024),
            optimizer=OptimizerConfig(learning_rate=3e-4),
            model_name="gpt2-medium"
        )
    
    @staticmethod
    def bert_base() -> ModelConfig:
        """BERT-Base配置 (110M参数)"""
        return ModelConfig(
            model_architecture=ModelArchitecture.BERT,
            transformer=TransformerConfig(
                vocab_size=30522, hidden_size=768, num_layers=12, num_heads=12,
                intermediate_size=3072, max_position_embeddings=512
            ),
            training=TrainingConfig(batch_size=32, max_seq_length=512),
            optimizer=OptimizerConfig(learning_rate=3e-5),
            model_name="bert-base-uncased"
        )
    
    @staticmethod
    def llama_7b() -> ModelConfig:
        """LLaMA-7B配置"""
        return ModelConfig(
            model_architecture=ModelArchitecture.LLAMA,
            transformer=TransformerConfig(
                vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32,
                intermediate_size=11008, max_position_embeddings=2048,
                use_rotary_embeddings=True, normalization_type=NormalizationType.RMS_NORM
            ),
            training=TrainingConfig(batch_size=4, max_seq_length=2048, gradient_checkpointing=True),
            optimizer=OptimizerConfig(learning_rate=1e-4),
            model_name="llama-7b"
        )
    
    @staticmethod
    def resnet50() -> ModelConfig:
        """ResNet-50配置"""
        return ModelConfig(
            model_architecture=ModelArchitecture.RESNET,
            cnn=CNNConfig(input_channels=3, num_classes=1000, num_layers=50, use_residual=True),
            training=TrainingConfig(batch_size=256),
            optimizer=OptimizerConfig(learning_rate=1e-3),
            model_name="resnet50"
        )


__all__ = [
    "ModelArchitecture", "ActivationFunction", "NormalizationType",
    "OptimizerType", "LRSchedulerType", "PrecisionType",
    "TransformerConfig", "CNNConfig", "RNNConfig",
    "OptimizerConfig", "LRSchedulerConfig", "TrainingConfig",
    "ModelConfig", "ModelPresets"
]
