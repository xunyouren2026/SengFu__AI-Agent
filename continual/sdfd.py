"""
慢速双网络防止漂移 (Slow-Dual-Network For Drift prevention, SDFT)

用于持续学习中防止策略漂移的技术。维护两个网络：
- 快速网络（在线网络）：用于日常学习和决策
- 慢速网络（目标网络）：用于提供稳定的目标，防止快速网络的过度适应

核心思想：慢速网络以较低的频率更新，作为"锚点"防止快速网络在新任务上过度优化而遗忘旧知识。
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UpdateMode(Enum):
    """慢速网络更新模式"""
    POLYAK = "polyak"  # Polyak平均：θ_slow = τ*θ_fast + (1-τ)*θ_slow
    PERIODIC = "periodic"  # 周期性复制：每隔N步完全复制
    EXPONENTIAL = "exponential"  # 指数衰减：更新频率随时间降低
    ADAPTIVE = "adaptive"  # 自适应：根据任务难度调整更新频率


@dataclass
class SDFTConfig:
    """SDFT配置"""
    # Polyak平均系数 (0, 1]，越小更新越慢
    tau: float = 0.005
    
    # 周期性更新间隔（步数）
    update_period: int = 1000
    
    # 更新模式
    update_mode: UpdateMode = UpdateMode.POLYAK
    
    # 是否启用EMA（指数移动平均）
    use_ema: bool = True
    
    # EMA衰减系数
    ema_decay: float = 0.999
    
    # 自适应更新的任务难度阈值
    difficulty_threshold: float = 0.7
    
    # 最小更新间隔（自适应模式）
    min_update_period: int = 100
    
    # 最大更新间隔（自适应模式）
    max_update_period: int = 10000
    
    # 是否同步BN统计量
    sync_bn: bool = True
    
    # 是否同步优化器状态
    sync_optimizer: bool = False


class SlowFastNetwork:
    """
    慢速-快速双网络管理器
    
    管理一对网络：快速网络用于学习和决策，慢速网络提供稳定目标。
    """
    
    def __init__(
        self,
        fast_network: nn.Module,
        config: Optional[SDFTConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化慢快双网络
        
        Args:
            fast_network: 快速网络（主网络）
            config: SDFT配置
            device: 计算设备
        """
        self.config = config or SDFTConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 快速网络（在线网络）
        self.fast_network = fast_network.to(self.device)
        
        # 慢速网络（目标网络）- 初始为快速网络的深拷贝
        self.slow_network = copy.deepcopy(fast_network).to(self.device)
        self.slow_network.eval()  # 慢速网络始终处于评估模式
        
        # 禁用慢速网络的梯度计算
        for param in self.slow_network.parameters():
            param.requires_grad = False
        
        # 更新计数器
        self.update_count = 0
        self.step_count = 0
        
        # 自适应模式下的动态更新间隔
        self.current_update_period = self.config.update_period
        
        # EMA状态（如果启用）
        self.ema_state: Dict[str, torch.Tensor] = {}
        if self.config.use_ema:
            self._init_ema()
        
        logger.info(f"SDFT初始化完成: mode={self.config.update_mode.value}, tau={self.config.tau}")
    
    def _init_ema(self):
        """初始化EMA状态"""
        self.ema_state = {
            name: param.clone().detach()
            for name, param in self.fast_network.named_parameters()
        }
    
    def update_slow_network(self, force: bool = False) -> Dict[str, Any]:
        """
        更新慢速网络
        
        Args:
            force: 强制更新，忽略更新间隔
            
        Returns:
            更新统计信息
        """
        self.step_count += 1
        
        # 检查是否需要更新
        if not force:
            if self.config.update_mode == UpdateMode.POLYAK:
                # Polyak模式：每步都更新，但使用小系数
                pass
            elif self.config.update_mode == UpdateMode.PERIODIC:
                if self.step_count % self.config.update_period != 0:
                    return {"updated": False, "mode": "periodic"}
            elif self.config.update_mode == UpdateMode.EXPONENTIAL:
                # 指数模式：更新间隔随时间指数增长
                if self.step_count % self.current_update_period != 0:
                    return {"updated": False, "mode": "exponential"}
                self.current_update_period = min(
                    int(self.current_update_period * 1.01),
                    self.config.max_update_period
                )
            elif self.config.update_mode == UpdateMode.ADAPTIVE:
                if self.step_count % self.current_update_period != 0:
                    return {"updated": False, "mode": "adaptive"}
        
        self.update_count += 1
        
        # 执行更新
        if self.config.update_mode == UpdateMode.POLYAK:
            update_info = self._polyak_update()
        elif self.config.update_mode in [UpdateMode.PERIODIC, UpdateMode.EXPONENTIAL]:
            update_info = self._periodic_update()
        elif self.config.update_mode == UpdateMode.ADAPTIVE:
            update_info = self._adaptive_update()
        else:
            raise ValueError(f"未知的更新模式: {self.config.update_mode}")
        
        # 同步BN统计量
        if self.config.sync_bn:
            self._sync_bn_stats()
        
        # 更新EMA
        if self.config.use_ema:
            self._update_ema()
        
        logger.debug(f"慢速网络已更新: step={self.step_count}, count={self.update_count}")
        
        return {
            "updated": True,
            "mode": self.config.update_mode.value,
            "step": self.step_count,
            "update_count": self.update_count,
            **update_info
        }
    
    def _polyak_update(self) -> Dict[str, Any]:
        """
        Polyak平均更新
        
        θ_slow = τ * θ_fast + (1 - τ) * θ_slow
        """
        tau = self.config.tau
        
        with torch.no_grad():
            for slow_param, fast_param in zip(
                self.slow_network.parameters(),
                self.fast_network.parameters()
            ):
                slow_param.data.mul_(1 - tau).add_(fast_param.data, alpha=tau)
        
        return {"tau": tau, "method": "polyak"}
    
    def _periodic_update(self) -> Dict[str, Any]:
        """
        周期性完全复制
        
        每隔N步，将快速网络的参数完全复制到慢速网络
        """
        self.slow_network.load_state_dict(self.fast_network.state_dict())
        
        return {
            "period": self.current_update_period,
            "method": "copy",
            "sync_bn": self.config.sync_bn
        }
    
    def _adaptive_update(self) -> Dict[str, Any]:
        """
        自适应更新
        
        根据任务难度动态调整更新频率
        """
        # 这里简化处理，实际应该根据任务难度评估
        self.slow_network.load_state_dict(self.fast_network.state_dict())
        
        return {
            "period": self.current_update_period,
            "method": "adaptive_copy",
            "difficulty_threshold": self.config.difficulty_threshold
        }
    
    def _sync_bn_stats(self):
        """同步BatchNorm统计量"""
        for slow_m, fast_m in zip(
            self.slow_network.modules(),
            self.fast_network.modules()
        ):
            if isinstance(slow_m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                slow_m.running_mean.copy_(fast_m.running_mean)
                slow_m.running_var.copy_(fast_m.running_var)
                slow_m.num_batches_tracked.copy_(fast_m.num_batches_tracked)
    
    def _update_ema(self):
        """更新EMA状态"""
        decay = self.config.ema_decay
        
        with torch.no_grad():
            for name, param in self.fast_network.named_parameters():
                if name in self.ema_state:
                    self.ema_state[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def get_slow_network(self) -> nn.Module:
        """获取慢速网络（用于推理或作为目标）"""
        return self.slow_network
    
    def get_fast_network(self) -> nn.Module:
        """获取快速网络（用于训练）"""
        return self.fast_network
    
    def get_ema_network(self) -> Optional[nn.Module]:
        """
        获取EMA网络
        
        如果启用了EMA，返回一个使用EMA参数的网络副本
        """
        if not self.config.use_ema:
            return None
        
        ema_network = copy.deepcopy(self.fast_network)
        with torch.no_grad():
            for name, param in ema_network.named_parameters():
                if name in self.ema_state:
                    param.data.copy_(self.ema_state[name])
        
        return ema_network
    
    def compute_target(self, *args, **kwargs) -> torch.Tensor:
        """
        使用慢速网络计算目标值
        
        这是SDFT的核心用途：提供稳定的学习目标
        """
        with torch.no_grad():
            return self.slow_network(*args, **kwargs)
    
    def compute_fast_output(self, *args, **kwargs) -> torch.Tensor:
        """使用快速网络计算输出"""
        return self.fast_network(*args, **kwargs)
    
    def get_drift_estimate(self) -> Dict[str, float]:
        """
        估计网络漂移程度
        
        计算快慢网络参数的差异，作为漂移的指标
        """
        total_diff = 0.0
        total_norm = 0.0
        param_diffs = {}
        
        with torch.no_grad():
            for (name, fast_param), (_, slow_param) in zip(
                self.fast_network.named_parameters(),
                self.slow_network.named_parameters()
            ):
                diff = (fast_param - slow_param).norm().item()
                norm = slow_param.norm().item()
                
                total_diff += diff ** 2
                total_norm += norm ** 2
                
                if norm > 0:
                    param_diffs[name] = diff / norm
        
        drift_ratio = (total_diff ** 0.5) / (total_norm ** 0.5 + 1e-8)
        
        return {
            "drift_ratio": drift_ratio,
            "total_diff": total_diff ** 0.5,
            "total_norm": total_norm ** 0.5,
            "max_param_diff": max(param_diffs.values()) if param_diffs else 0.0,
            "mean_param_diff": sum(param_diffs.values()) / len(param_diffs) if param_diffs else 0.0,
            "update_count": self.update_count,
            "step_count": self.step_count
        }
    
    def adjust_update_frequency(self, task_difficulty: float):
        """
        自适应调整更新频率
        
        Args:
            task_difficulty: 任务难度评分 [0, 1]
        """
        if self.config.update_mode != UpdateMode.ADAPTIVE:
            return
        
        # 任务越难，更新越频繁（间隔越小）
        if task_difficulty > self.config.difficulty_threshold:
            # 困难任务：更频繁更新
            self.current_update_period = max(
                self.current_update_period * 0.9,
                self.config.min_update_period
            )
        else:
            # 简单任务：减少更新频率
            self.current_update_period = min(
                self.current_update_period * 1.1,
                self.config.max_update_period
            )
        
        logger.info(f"更新频率调整: difficulty={task_difficulty:.3f}, period={self.current_update_period}")
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            "fast_state_dict": self.fast_network.state_dict(),
            "slow_state_dict": self.slow_network.state_dict(),
            "config": self.config,
            "update_count": self.update_count,
            "step_count": self.step_count,
            "current_update_period": self.current_update_period,
            "ema_state": self.ema_state if self.config.use_ema else None
        }
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.fast_network.load_state_dict(checkpoint["fast_state_dict"])
        self.slow_network.load_state_dict(checkpoint["slow_state_dict"])
        self.update_count = checkpoint["update_count"]
        self.step_count = checkpoint["step_count"]
        self.current_update_period = checkpoint.get("current_update_period", self.config.update_period)
        
        if self.config.use_ema and "ema_state" in checkpoint and checkpoint["ema_state"]:
            self.ema_state = checkpoint["ema_state"]
        
        logger.info(f"检查点已加载: {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        drift = self.get_drift_estimate()
        
        return {
            "update_count": self.update_count,
            "step_count": self.step_count,
            "update_frequency": self.update_count / max(self.step_count, 1),
            "current_update_period": self.current_update_period,
            **drift,
            "config": {
                "tau": self.config.tau,
                "update_mode": self.config.update_mode.value,
                "use_ema": self.config.use_ema
            }
        }


class SDFTTrainer:
    """
    使用SDFT的训练器包装器
    
    简化在标准训练循环中使用SDFT的过程
    """
    
    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[SDFTConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化SDFT训练器
        
        Args:
            network: 要训练的网络
            optimizer: 优化器
            config: SDFT配置
            device: 计算设备
        """
        self.sdft = SlowFastNetwork(network, config, device)
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_losses = []
        self.drift_ratios = []
    
    def training_step(
        self,
        batch: tuple,
        loss_fn: Callable,
        use_slow_target: bool = True
    ) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            batch: (inputs, targets)元组
            loss_fn: 损失函数
            use_slow_target: 是否使用慢速网络计算目标
            
        Returns:
            训练统计
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        self.sdft.fast_network.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.sdft.compute_fast_output(inputs)
        
        # 计算损失
        if use_slow_target:
            # 使用慢速网络提供的目标（例如DQN中的目标Q值）
            with torch.no_grad():
                target_outputs = self.sdft.compute_target(inputs)
            loss = loss_fn(outputs, targets, target_outputs)
        else:
            loss = loss_fn(outputs, targets)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        # 更新慢速网络
        update_info = self.sdft.update_slow_network()
        
        # 记录统计
        loss_val = loss.item()
        self.train_losses.append(loss_val)
        
        drift = self.sdft.get_drift_estimate()
        self.drift_ratios.append(drift["drift_ratio"])
        
        return {
            "loss": loss_val,
            "drift_ratio": drift["drift_ratio"],
            "slow_updated": update_info.get("updated", False),
            **update_info
        }
    
    def get_network(self, use_slow: bool = False) -> nn.Module:
        """获取网络（快速或慢速）"""
        if use_slow:
            return self.sdft.get_slow_network()
        return self.sdft.get_fast_network()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        import numpy as np
        
        sdft_stats = self.sdft.get_stats()
        
        return {
            **sdft_stats,
            "mean_loss": np.mean(self.train_losses) if self.train_losses else 0.0,
            "mean_drift": np.mean(self.drift_ratios) if self.drift_ratios else 0.0,
            "max_drift": max(self.drift_ratios) if self.drift_ratios else 0.0
        }


def create_sdft_optimizer(
    network: nn.Module,
    base_optimizer_class: type,
    config: Optional[SDFTConfig] = None,
    **optimizer_kwargs
) -> SDFTTrainer:
    """
    工厂函数：创建带有SDFT的训练器
    
    Args:
        network: 网络模型
        base_optimizer_class: 基础优化器类（如torch.optim.Adam）
        config: SDFT配置
        **optimizer_kwargs: 优化器参数
        
    Returns:
        SDFTTrainer实例
    """
    optimizer = base_optimizer_class(network.parameters(), **optimizer_kwargs)
    return SDFTTrainer(network, optimizer, config)


# 兼容性别名
SlowDualNetwork = SlowFastNetwork
