"""
Parseval调度器 - Parseval Scheduler

基于Parseval框架的网络Lipschitz常数控制模块。

核心原理：
  Parseval定理保证正交基下信号能量守恒。将此约束推广到神经网络：
  - 每层的权重矩阵近似正交（Parseval约束）
  - 保证整体网络的Lipschitz常数有界
  - 防止梯度爆炸/消失，提升训练稳定性

数学基础：
  对权重矩阵W，Parseval约束要求 W^T W ≈ I
  即权重矩阵的列向量近似正交归一。

调度策略：
  1. 正交化：定期将权重投影到Stiefel流形上
  2. 谱归一化：通过SVD分解约束最大奇异值
  3. 自适应调度：根据训练阶段动态调整约束强度
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union
import math


class ParsevalScheduler:
    """
    Parseval调度器

    控制神经网络各层的Lipschitz常数，保证训练稳定性和泛化能力。
    通过正交化、谱归一化和自适应调度三种机制的组合，
    在保持网络表达能力的同时约束其Lipschitz性质。

    使用方式：
        scheduler = ParsevalScheduler(model, target_lipschitz=1.0)
        for epoch in range(num_epochs):
            scheduler.step(model, epoch)
            # 训练代码...
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        target_lipschitz: float = 1.0,
        ortho_interval: int = 100,
        spectral_norm_interval: int = 1,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        max_lipschitz: float = 2.0,
        min_lipschitz: float = 0.1,
        ortho_strength: float = 1.0,
        spectral_strength: float = 0.01,
        use_adaptive: bool = True,
        track_lipschitz: bool = True,
        verbose: bool = False,
    ):
        """
        初始化Parseval调度器

        Args:
            model: 需要约束的模型（可选，后续可通过set_model设置）
            target_lipschitz: 目标Lipschitz常数
            ortho_interval: 正交化操作的间隔步数
            spectral_norm_interval: 谱归一化操作的间隔步数
            warmup_epochs: 预热轮数（预热期内不施加约束）
            cooldown_epochs: 冷却轮数（训练末期逐渐放松约束）
            max_lipschitz: Lipschitz常数上限
            min_lipschitz: Lipschitz常数下限
            ortho_strength: 正交化强度（0~1，1为完全正交化）
            spectral_strength: 谱归一化强度（0~1）
            use_adaptive: 是否启用自适应调度
            track_lipschitz: 是否跟踪Lipschitz常数变化
            verbose: 是否打印调度信息
        """
        self.model = model
        self.target_lipschitz = target_lipschitz
        self.ortho_interval = ortho_interval
        self.spectral_norm_interval = spectral_norm_interval
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.max_lipschitz = max_lipschitz
        self.min_lipschitz = min_lipschitz
        self.ortho_strength = ortho_strength
        self.spectral_strength = spectral_strength
        self.use_adaptive = use_adaptive
        self.track_lipschitz = track_lipschitz
        self.verbose = verbose

        # ---- 内部状态 ----
        self.current_step = 0
        self.current_epoch = 0
        self.total_epochs = 0

        # Lipschitz常数跟踪
        self.lipschitz_history: List[float] = []
        self.layer_lipschitz_history: Dict[str, List[float]] = {}

        # 自适应调度的参数
        self.adaptive_lipschitz_target = target_lipschitz
        self.lipschitz_moving_avg = target_lipschitz
        self.lipschitz_momentum = 0.99

    def set_model(self, model: nn.Module):
        """
        设置需要约束的模型

        Args:
            model: PyTorch模型
        """
        self.model = model

    def set_total_epochs(self, total_epochs: int):
        """
        设置总训练轮数（用于自适应调度）

        Args:
            total_epochs: 总训练轮数
        """
        self.total_epochs = total_epochs

    # ================================================================
    #  核心调度方法
    # ================================================================

    def step(
        self,
        model: Optional[nn.Module] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        执行一步调度

        根据当前训练阶段，选择性地执行正交化和/或谱归一化。

        Args:
            model: 模型（可选，默认使用构造时传入的模型）
            epoch: 当前轮数（可选）
            step: 当前步数（可选）

        Returns:
            调度统计信息字典
        """
        if model is not None:
            self.model = model
        if epoch is not None:
            self.current_epoch = epoch
        if step is not None:
            self.current_step = step

        assert self.model is not None, "必须提供模型"

        stats = {}

        # ---- 预热期检查 ----
        if self.current_epoch < self.warmup_epochs:
            if self.verbose:
                print(f"[ParsevalScheduler] 预热期 (epoch {self.current_epoch}), "
                      f"跳过约束")
            stats['phase'] = 'warmup'
            stats['ortho_applied'] = False
            stats['spectral_applied'] = False
            return stats

        # ---- 冷却期检查 ----
        if self.total_epochs > 0:
            remaining = self.total_epochs - self.current_epoch
            if remaining <= self.cooldown_epochs:
                # 冷却期：逐渐放松约束
                cooldown_factor = remaining / self.cooldown_epochs
                effective_ortho = self.ortho_strength * cooldown_factor
                effective_spectral = self.spectral_strength * cooldown_factor
                stats['phase'] = 'cooldown'
                stats['cooldown_factor'] = cooldown_factor
            else:
                effective_ortho = self.ortho_strength
                effective_spectral = self.spectral_strength
                stats['phase'] = 'active'
        else:
            effective_ortho = self.ortho_strength
            effective_spectral = self.spectral_strength
            stats['phase'] = 'active'

        # ---- 自适应调度 ----
        if self.use_adaptive:
            self._adaptive_update()
            stats['adaptive_target'] = self.adaptive_lipschitz_target

        # ---- 正交化 ----
        ortho_applied = False
        if self.ortho_interval > 0 and self.current_step % self.ortho_interval == 0:
            if effective_ortho > 0:
                self._orthogonalize_weights(strength=effective_ortho)
                ortho_applied = True
        stats['ortho_applied'] = ortho_applied

        # ---- 谱归一化 ----
        spectral_applied = False
        if (self.spectral_norm_interval > 0 and
                self.current_step % self.spectral_norm_interval == 0):
            if effective_spectral > 0:
                self._apply_spectral_normalization(strength=effective_spectral)
                spectral_applied = True
        stats['spectral_applied'] = spectral_applied

        # ---- 跟踪Lipschitz常数 ----
        if self.track_lipschitz and (spectral_applied or ortho_applied):
            lipschitz = self.estimate_lipschitz()
            self.lipschitz_history.append(lipschitz)
            stats['estimated_lipschitz'] = lipschitz

        self.current_step += 1

        return stats

    # ================================================================
    #  正交化
    # ================================================================

    def _orthogonalize_weights(self, strength: float = 1.0):
        """
        对模型权重执行正交化

        将每个权重矩阵投影到Stiefel流形上（或部分投影），
        使得 W^T W ≈ I。

        使用Cayley变换实现平滑的正交化：
          W_ortho = W * (I + α/2 * (W^T W - I))^(-1)

        其中 α 控制正交化强度。

        Args:
            strength: 正交化强度 (0~1)
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2:
                continue

            # 只对2D权重矩阵执行正交化
            if param.dim() == 2:
                with torch.no_grad():
                    # 使用Cayley变换进行平滑正交化
                    W = param.data
                    rows, cols = W.shape

                    if rows >= cols:
                        # 高瘦矩阵：右正交化 W^T W ≈ I
                        WtW = W.t() @ W
                        I = torch.eye(cols, device=W.device, dtype=W.dtype)
                        diff = WtW - I
                        # Cayley变换：(I + α/2 * diff)^(-1)
                        alpha = strength * 0.5
                        try:
                            correction = torch.linalg.solve(
                                I + alpha * diff, I
                            )
                            param.data = W @ correction
                        except torch.linalg.LinAlgError:
                            # 如果矩阵奇异，使用SVD正交化
                            param.data = self._svd_orthogonalize(W, strength)
                    else:
                        # 矮胖矩阵：左正交化 W W^T ≈ I
                        WWt = W @ W.t()
                        I = torch.eye(rows, device=W.device, dtype=W.dtype)
                        diff = WWt - I
                        alpha = strength * 0.5
                        try:
                            correction = torch.linalg.solve(
                                I + alpha * diff, I
                            )
                            param.data = correction @ W
                        except torch.linalg.LinAlgError:
                            param.data = self._svd_orthogonalize(W, strength)

    def _svd_orthogonalize(
        self,
        W: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        使用SVD分解进行正交化

        将权重矩阵分解为 W = U * S * V^T，
        然后将奇异值向1收缩：S' = (1-α)*S + α*1

        Args:
            W: 权重矩阵
            strength: 正交化强度

        Returns:
            正交化后的权重矩阵
        """
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            # 将奇异值向1收缩
            S_new = (1 - strength) * S + strength * torch.ones_like(S)

            # 重建矩阵
            return U @ torch.diag(S_new) @ Vh
        except Exception:
            return W

    def orthogonalize_layer(
        self,
        layer: nn.Module,
        strength: float = 1.0,
    ):
        """
        对单个层执行正交化

        Args:
            layer: PyTorch层
            strength: 正交化强度
        """
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                W = layer.weight.data
                layer.weight.data = self._svd_orthogonalize(W, strength)
        elif isinstance(layer, nn.Conv2d):
            with torch.no_grad():
                # 将卷积核重塑为2D矩阵进行正交化
                W = layer.weight.data
                shape = W.shape
                W_2d = W.reshape(shape[0], -1)
                W_ortho = self._svd_orthogonalize(W_2d, strength)
                layer.weight.data = W_ortho.reshape(shape)

    # ================================================================
    #  谱归一化
    # ================================================================

    def _apply_spectral_normalization(self, strength: float = 0.01):
        """
        对模型权重应用谱归一化

        谱归一化将权重矩阵的最大奇异值约束为目标值：
          W_normalized = W / σ_max(W) * target

        其中 σ_max(W) 是W的最大奇异值，target是目标谱范数。

        Args:
            strength: 谱归一化强度（控制更新幅度）
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2:
                continue

            with torch.no_grad():
                if param.dim() == 2:
                    # 计算最大奇异值（幂迭代近似）
                    sigma_max = self._estimate_max_singular_value(param.data)

                    if sigma_max > 1e-8:
                        # 计算缩放因子
                        target_sigma = self.adaptive_lipschitz_target
                        scale = target_sigma / sigma_max

                        # 平滑更新：W_new = (1-α)*W + α*(W * scale)
                        param.data = (
                            (1 - strength) * param.data +
                            strength * param.data * scale
                        )

    def _estimate_max_singular_value(
        self,
        W: torch.Tensor,
        num_iterations: int = 3,
    ) -> float:
        """
        使用幂迭代法估计最大奇异值

        幂迭代法通过反复乘以 W^T W 来逼近最大特征值，
        其平方根即为最大奇异值。

        算法复杂度：O(num_iterations * nnz(W))，非常高效。

        Args:
            W: 权重矩阵
            num_iterations: 迭代次数

        Returns:
            估计的最大奇异值
        """
        rows, cols = W.shape

        # 初始化随机向量
        v = torch.randn(cols, device=W.device, dtype=W.dtype)
        v = v / (torch.norm(v) + 1e-8)

        for _ in range(num_iterations):
            # v_new = W^T W v / ||W^T W v||
            Wv = W @ v
            WtWv = W.t() @ Wv
            norm = torch.norm(WtWv)
            if norm > 1e-8:
                v = WtWv / norm
            else:
                break

        # 最大奇异值 = sqrt(v^T W^T W v)
        Wv = W @ v
        sigma = torch.norm(Wv)

        return sigma.item()

    def compute_spectral_norm(self, W: torch.Tensor) -> float:
        """
        计算权重矩阵的精确谱范数

        使用完整的SVD分解计算最大奇异值。
        注意：此方法计算量较大，仅用于验证。

        Args:
            W: 权重矩阵

        Returns:
            精确的谱范数（最大奇异值）
        """
        try:
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            return S[0].item()
        except Exception:
            return self._estimate_max_singular_value(W, num_iterations=10)

    # ================================================================
    #  自适应调度
    # ================================================================

    def _adaptive_update(self):
        """
        自适应调整Lipschitz常数目标

        根据训练过程中Lipschitz常数的实际变化趋势，
        动态调整目标值：
        - 如果Lipschitz常数持续增长，降低目标值
        - 如果Lipschitz常数稳定或下降，保持或略微提高目标值

        这允许网络在训练初期有更大的表达能力，
        而在训练后期更加稳定。
        """
        if len(self.lipschitz_history) < 5:
            return

        # 计算最近的Lipschitz常数趋势
        recent = self.lipschitz_history[-5:]
        trend = recent[-1] - recent[0]

        # 更新移动平均
        current_lipschitz = recent[-1]
        self.lipschitz_moving_avg = (
            self.lipschitz_momentum * self.lipschitz_moving_avg +
            (1 - self.lipschitz_momentum) * current_lipschitz
        )

        # 根据趋势调整目标
        if trend > 0.1:
            # Lipschitz常数在增长，降低目标
            self.adaptive_lipschitz_target *= 0.95
        elif trend < -0.1:
            # Lipschitz常数在下降，可以略微放松
            self.adaptive_lipschitz_target *= 1.02

        # 限制在合理范围内
        self.adaptive_lipschitz_target = max(
            self.min_lipschitz,
            min(self.max_lipschitz, self.adaptive_lipschitz_target)
        )

    def get_schedule_factor(self, epoch: int) -> float:
        """
        获取指定轮次的调度因子

        调度因子控制约束强度：
        - 预热期：0（无约束）
        - 活跃期：1（完全约束）
        - 冷却期：线性衰减到0

        Args:
            epoch: 当前轮数

        Returns:
            调度因子 [0, 1]
        """
        if epoch < self.warmup_epochs:
            return 0.0

        if self.total_epochs > 0:
            remaining = self.total_epochs - epoch
            if remaining <= self.cooldown_epochs:
                return remaining / self.cooldown_epochs

        return 1.0

    # ================================================================
    #  Lipschitz常数估计
    # ================================================================

    def estimate_lipschitz(
        self,
        model: Optional[nn.Module] = None,
        num_samples: int = 10,
        input_dim: Optional[int] = None,
    ) -> float:
        """
        估计模型的Lipschitz常数

        使用随机输入通过幂迭代法估计整体Lipschitz常数。
        Lipschitz常数 L 满足：||f(x) - f(y)|| <= L * ||x - y||

        估计方法：L ≈ max_{||x||=1} ||J_f(x)||_2
        通过随机采样和Jacobian范数来近似。

        Args:
            model: 模型（可选）
            num_samples: 随机采样次数
            input_dim: 输入维度（可选，自动推断）

        Returns:
            估计的Lipschitz常数
        """
        if model is not None:
            self.model = model
        assert self.model is not None

        self.model.eval()
        device = next(self.model.parameters()).device

        # 推断输入维度
        if input_dim is None:
            # 尝试从第一层推断
            first_param = next(self.model.parameters())
            input_dim = first_param.shape[1] if first_param.dim() >= 2 else 64

        max_lipschitz = 0.0

        with torch.no_grad():
            for _ in range(num_samples):
                # 生成随机单位向量
                x = torch.randn(1, input_dim, device=device)
                x = x / (torch.norm(x) + 1e-8)

                # 前向传播
                try:
                    output = self.model(x)

                    # 计算输出范数作为Lipschitz常数的下界估计
                    output_norm = torch.norm(output)
                    max_lipschitz = max(max_lipschitz, output_norm.item())
                except Exception:
                    # 如果前向传播失败，跳过
                    continue

        # 逐层估计（更精确）
        layer_lipschitz = self._estimate_per_layer_lipschitz()
        if layer_lipschitz > 0:
            # 取逐层估计和整体估计的较大值
            max_lipschitz = max(max_lipschitz, layer_lipschitz)

        self.model.train()
        return max_lipschitz

    def _estimate_per_layer_lipschitz(self) -> float:
        """
        逐层估计Lipschitz常数

        整体Lipschitz常数 <= 各层Lipschitz常数之积。
        这里返回乘积的对数，避免数值溢出。

        Returns:
            逐层Lipschitz常数乘积
        """
        total_log_lipschitz = 0.0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2:
                continue

            with torch.no_grad():
                sigma_max = self._estimate_max_singular_value(param.data, 5)
                if sigma_max > 1e-8:
                    total_log_lipschitz += math.log(sigma_max)

                    # 记录每层的Lipschitz常数
                    if self.track_lipschitz:
                        if name not in self.layer_lipschitz_history:
                            self.layer_lipschitz_history[name] = []
                        self.layer_lipschitz_history[name].append(sigma_max)

        return math.exp(total_log_lipschitz)

    # ================================================================
    #  工具方法
    # ================================================================

    def get_lipschitz_history(self) -> List[float]:
        """
        获取Lipschitz常数历史记录

        Returns:
            Lipschitz常数列表
        """
        return self.lipschitz_history.copy()

    def get_layer_lipschitz_report(self) -> Dict[str, Dict[str, float]]:
        """
        获取各层Lipschitz常数报告

        Returns:
            每层的Lipschitz常数统计信息
        """
        report = {}
        for name, history in self.layer_lipschitz_history.items():
            if len(history) > 0:
                report[name] = {
                    'current': history[-1],
                    'mean': sum(history) / len(history),
                    'max': max(history),
                    'min': min(history),
                    'num_updates': len(history),
                }
        return report

    def get_parseval_violation(
        self,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        计算Parseval约束违反程度

        对每个权重矩阵计算 ||W^T W - I||_F，
        衡量其偏离正交性的程度。

        Args:
            model: 模型（可选）

        Returns:
            各层Parseval违反程度的字典
        """
        if model is not None:
            self.model = model
        assert self.model is not None

        violations = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() != 2:
                continue

            with torch.no_grad():
                W = param.data
                rows, cols = W.shape

                if rows >= cols:
                    WtW = W.t() @ W
                    I = torch.eye(cols, device=W.device, dtype=W.dtype)
                else:
                    WtW = W @ W.t()
                    I = torch.eye(rows, device=W.device, dtype=W.dtype)

                violation = torch.norm(WtW - I, p='fro').item()
                violations[name] = violation

        return violations

    def reset(self):
        """
        重置调度器状态

        清空所有历史记录，恢复初始参数。
        """
        self.current_step = 0
        self.current_epoch = 0
        self.lipschitz_history.clear()
        self.layer_lipschitz_history.clear()
        self.adaptive_lipschitz_target = self.target_lipschitz
        self.lipschitz_moving_avg = self.target_lipschitz

    def state_dict(self) -> Dict:
        """
        获取调度器状态

        Returns:
            状态字典
        """
        return {
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'adaptive_lipschitz_target': self.adaptive_lipschitz_target,
            'lipschitz_moving_avg': self.lipschitz_moving_avg,
            'lipschitz_history': self.lipschitz_history,
        }

    def load_state_dict(self, state_dict: Dict):
        """
        加载调度器状态

        Args:
            state_dict: 状态字典
        """
        self.current_step = state_dict.get('current_step', 0)
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.total_epochs = state_dict.get('total_epochs', 0)
        self.adaptive_lipschitz_target = state_dict.get(
            'adaptive_lipschitz_target', self.target_lipschitz
        )
        self.lipschitz_moving_avg = state_dict.get(
            'lipschitz_moving_avg', self.target_lipschitz
        )
        self.lipschitz_history = state_dict.get('lipschitz_history', [])

    def __repr__(self) -> str:
        return (
            f'ParsevalScheduler('
            f'target_L={self.target_lipschitz}, '
            f'ortho_interval={self.ortho_interval}, '
            f'spectral_interval={self.spectral_norm_interval}, '
            f'warmup={self.warmup_epochs}, '
            f'adaptive={self.use_adaptive})'
        )
