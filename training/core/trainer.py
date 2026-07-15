"""
训练基础设施 - Training Infrastructure

提供完整的训练支持：
- 训练循环管理
- 混合精度训练
- 梯度累积
- 学习率调度
- 早停机制
- 检查点管理
- 分布式训练支持
- 日志和监控
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Callable, Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import logging
from enum import Enum
import math


class LRSchedulerType(Enum):
    """学习率调度器类型"""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    WARMUP_COSINE = "warmup_cosine"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    max_epochs: int = 100
    max_steps: Optional[int] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # 学习率
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-7
    weight_decay: float = 0.01
    lr_scheduler: LRSchedulerType = LRSchedulerType.WARMUP_COSINE
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    
    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16, bfloat16
    
    # 梯度
    max_grad_norm: float = 1.0
    grad_clip_type: str = "norm"  # norm, value
    
    # 早停
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # 检查点
    checkpoint_dir: str = "./checkpoints"
    save_every_n_steps: int = 1000
    save_every_n_epochs: int = 1
    keep_n_checkpoints: int = 5
    
    # 日志
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 500
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 分布式
    distributed: bool = False
    local_rank: int = -1


class LRScheduler:
    """学习率调度器"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        config: TrainingConfig,
        num_training_steps: int
    ):
        self.optimizer = optimizer
        self.config = config
        self.num_training_steps = num_training_steps
        
        self.current_step = 0
        self.base_lr = config.learning_rate
        
        # 计算warmup步数
        if config.warmup_ratio > 0:
            self.warmup_steps = int(num_training_steps * config.warmup_ratio)
        else:
            self.warmup_steps = config.warmup_steps
    
    def step(self) -> float:
        """更新学习率"""
        self.current_step += 1
        
        if self.config.lr_scheduler == LRSchedulerType.CONSTANT:
            lr = self.base_lr
        
        elif self.config.lr_scheduler == LRSchedulerType.LINEAR:
            if self.current_step < self.warmup_steps:
                lr = self.base_lr * self.current_step / self.warmup_steps
            else:
                progress = (self.current_step - self.warmup_steps) / \
                          (self.num_training_steps - self.warmup_steps)
                lr = self.base_lr * (1 - progress)
        
        elif self.config.lr_scheduler == LRSchedulerType.COSINE:
            if self.current_step < self.warmup_steps:
                lr = self.base_lr * self.current_step / self.warmup_steps
            else:
                progress = (self.current_step - self.warmup_steps) / \
                          (self.num_training_steps - self.warmup_steps)
                lr = self.config.min_learning_rate + \
                    (self.base_lr - self.config.min_learning_rate) * \
                    0.5 * (1 + math.cos(math.pi * progress))
        
        elif self.config.lr_scheduler == LRSchedulerType.WARMUP_COSINE:
            if self.current_step < self.warmup_steps:
                lr = self.base_lr * self.current_step / self.warmup_steps
            else:
                progress = (self.current_step - self.warmup_steps) / \
                          max(1, self.num_training_steps - self.warmup_steps)
                lr = self.config.min_learning_rate + \
                    (self.base_lr - self.config.min_learning_rate) * \
                    0.5 * (1 + math.cos(math.pi * progress))
        
        elif self.config.lr_scheduler == LRSchedulerType.EXPONENTIAL:
            decay_rate = math.log(self.config.min_learning_rate / self.base_lr) / \
                        self.num_training_steps
            lr = self.base_lr * math.exp(decay_rate * self.current_step)
        
        else:
            lr = self.base_lr
        
        # 更新优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """检查是否应该停止"""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class CheckpointManager:
    """检查点管理"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_n_checkpoints: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_checkpoints = keep_n_checkpoints
        
        self.checkpoints: List[Tuple[str, int, float]] = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[LRScheduler],
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        extra_state: Optional[Dict] = None
    ) -> str:
        """保存检查点"""
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_step'] = scheduler.current_step
        
        if extra_state:
            checkpoint['extra_state'] = extra_state
        
        # 保存
        filename = f"checkpoint_step{step}_epoch{epoch}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        # 记录
        score = metrics.get('val_loss', metrics.get('loss', float('inf')))
        self.checkpoints.append((str(filepath), step, score))
        
        # 清理旧检查点
        self._cleanup()
        
        return str(filepath)
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """加载检查点"""
        if checkpoint_path is None:
            # 找最新的
            if not self.checkpoints:
                return {}
            checkpoint_path = self.checkpoints[-1][0]
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_step' in checkpoint:
            scheduler.current_step = checkpoint['scheduler_step']
        
        return checkpoint
    
    def load_best(self, model: nn.Module) -> Optional[Dict]:
        """加载最佳检查点"""
        if not self.checkpoints:
            return None
        
        # 找loss最小的
        best = min(self.checkpoints, key=lambda x: x[2])
        return self.load(model, checkpoint_path=best[0])
    
    def _cleanup(self) -> None:
        """清理旧检查点"""
        if len(self.checkpoints) <= self.keep_n_checkpoints:
            return
        
        # 按step排序
        self.checkpoints.sort(key=lambda x: x[1])
        
        # 删除最旧的
        while len(self.checkpoints) > self.keep_n_checkpoints:
            old_path, _, _ = self.checkpoints.pop(0)
            Path(old_path).unlink(missing_ok=True)


class Trainer:
    """
    训练器
    
    提供完整的训练循环。
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        eval_fn: Optional[Callable] = None
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        
        # 优化器
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # 计算训练步数
        num_epochs = config.max_epochs
        steps_per_epoch = len(train_dataloader)
        self.num_training_steps = num_epochs * steps_per_epoch
        if config.max_steps:
            self.num_training_steps = min(self.num_training_steps, config.max_steps)
        
        # 学习率调度器
        self.scheduler = LRScheduler(
            self.optimizer, config, self.num_training_steps
        )
        
        # 混合精度
        self.scaler = GradScaler() if config.use_amp else None
        self.amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
        
        # 早停
        self.early_stopping = EarlyStopping(
            config.early_stopping_patience,
            config.early_stopping_min_delta
        ) if config.early_stopping else None
        
        # 检查点
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            config.keep_n_checkpoints
        )
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        
        # 指标历史
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': []
        }
    
    def train(self) -> Dict[str, Any]:
        """
        执行训练
        
        Returns:
            训练结果
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Total steps: {self.num_training_steps}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch
                
                # 训练一个epoch
                train_metrics = self._train_epoch()
                
                # 评估
                eval_metrics = {}
                if self.eval_dataloader is not None:
                    eval_metrics = self._evaluate()
                
                # 记录
                self.history['train_loss'].append(train_metrics.get('loss', 0))
                self.history['eval_loss'].append(eval_metrics.get('loss', 0))
                self.history['learning_rate'].append(self.scheduler.get_lr())
                
                # 检查点
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self.checkpoint_manager.save(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, epoch,
                        {**train_metrics, **eval_metrics}
                    )
                
                # 早停
                if self.early_stopping:
                    eval_loss = eval_metrics.get('loss', train_metrics.get('loss', 0))
                    if self.early_stopping(eval_loss):
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                # 日志
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics.get('loss', 0):.4f}, "
                    f"eval_loss={eval_metrics.get('loss', 0):.4f}, "
                    f"lr={self.scheduler.get_lr():.6f}"
                )
                
                # 检查最大步数
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        elapsed = time.time() - start_time
        
        return {
            'history': self.history,
            'best_metric': self.best_metric,
            'total_steps': self.global_step,
            'total_epochs': self.current_epoch + 1,
            'elapsed_time': elapsed,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_eval_loss': self.history['eval_loss'][-1] if self.history['eval_loss'] else None
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 前向传播
            loss = self._forward_step(batch)
            
            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps
            self._backward_step(loss)
            
            # 更新
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self._optimizer_step()
                self.global_step += 1
                
                # 学习率调度
                self.scheduler.step()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # 日志
            if self.global_step % self.config.log_every_n_steps == 0:
                self.logger.debug(
                    f"Step {self.global_step}: loss={loss.item():.4f}, "
                    f"lr={self.scheduler.get_lr():.6f}"
                )
            
            # 保存检查点
            if self.global_step % self.config.save_every_n_steps == 0:
                self.checkpoint_manager.save(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step, self.current_epoch,
                    {'loss': total_loss / num_batches}
                )
            
            # 最大步数检查
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        return {'loss': total_loss / max(1, num_batches)}
    
    def _forward_step(self, batch: Any) -> torch.Tensor:
        """前向传播"""
        # 移动数据到设备
        if isinstance(batch, dict):
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            batch = [v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for v in batch]
        
        # 混合精度
        if self.config.use_amp:
            with autocast(dtype=self.amp_dtype):
                if self.loss_fn:
                    loss = self.loss_fn(self.model, batch)
                else:
                    outputs = self.model(**batch) if isinstance(batch, dict) else self.model(*batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        else:
            if self.loss_fn:
                loss = self.loss_fn(self.model, batch)
            else:
                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(*batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        
        return loss
    
    def _backward_step(self, loss: torch.Tensor) -> None:
        """反向传播"""
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self) -> None:
        """优化器更新"""
        # 梯度裁剪
        if self.config.max_grad_norm > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            if self.config.grad_clip_type == "norm":
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
        
        # 优化器步进
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def _evaluate(self) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # 移动数据
                if isinstance(batch, dict):
                    batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = [v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                            for v in batch]
                
                # 前向
                if self.config.use_amp:
                    with autocast(dtype=self.amp_dtype):
                        outputs = self.model(**batch) if isinstance(batch, dict) else self.model(*batch)
                else:
                    outputs = self.model(**batch) if isinstance(batch, dict) else self.model(*batch)
                
                # 收集指标
                if isinstance(outputs, dict):
                    loss = outputs.get('loss', 0)
                    total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                    
                    for k, v in outputs.items():
                        if k != 'loss' and isinstance(v, torch.Tensor):
                            total_metrics[k] = total_metrics.get(k, 0) + v.item()
                else:
                    total_loss += outputs.item() if isinstance(outputs, torch.Tensor) else outputs
                
                num_batches += 1
        
        # 平均
        metrics = {'loss': total_loss / max(1, num_batches)}
        for k, v in total_metrics.items():
            metrics[k] = v / max(1, num_batches)
        
        return metrics
    
    def resume(self, checkpoint_path: Optional[str] = None) -> None:
        """从检查点恢复"""
        state = self.checkpoint_manager.load(
            self.model, self.optimizer, self.scheduler, checkpoint_path
        )
        
        if state:
            self.global_step = state.get('step', 0)
            self.current_epoch = state.get('epoch', 0)
            self.logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
