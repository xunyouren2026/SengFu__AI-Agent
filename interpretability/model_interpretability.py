"""
模型解释性模块 - Model Interpretability
实现注意力可视化、特征归因、概念激活向量等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict

# ==================== 显著性方法 ====================

class SaliencyMap:
    """显著性图"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """计算显著性图"""
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=-1)
        
        if output.dim() > 1:
            score = output[:, target_class].sum()
        else:
            score = output[target_class].sum()
        
        score.backward()
        
        saliency = input_tensor.grad.abs()
        
        return saliency


class IntegratedGradients:
    """积分梯度"""
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50,
    ):
        self.model = model
        self.num_steps = num_steps
        self.model.eval()
    
    def compute(
        self,
        input_tensor: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """计算积分梯度"""
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # 插值路径
        alphas = torch.linspace(0, 1, self.num_steps + 1, device=input_tensor.device)
        
        # 计算梯度积分
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas[:-1]:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.clone().requires_grad_(True)
            
            output = self.model(interpolated)
            
            if target_class is None:
                target_class = output.argmax(dim=-1)
            
            if output.dim() > 1:
                score = output[:, target_class].sum()
            else:
                score = output[target_class].sum()
            
            score.backward()
            
            integrated_grads += interpolated.grad
        
        # 平均并缩放
        integrated_grads = integrated_grads / self.num_steps
        integrated_grads = integrated_grads * (input_tensor - baseline)
        
        return integrated_grads


class GradCAM:
    """Grad-CAM"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
        
        self.model.eval()
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """计算Grad-CAM"""
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=-1)
        
        if output.dim() > 1:
            score = output[:, target_class].sum()
        else:
            score = output[target_class].sum()
        
        self.model.zero_grad()
        score.backward()
        
        # 计算权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # 加权组合
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 归一化
        cam = cam / (cam.max() + 1e-8)
        
        return cam


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++"""
    
    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """计算Grad-CAM++"""
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=-1)
        
        if output.dim() > 1:
            score = output[:, target_class].sum()
        else:
            score = output[target_class].sum()
        
        self.model.zero_grad()
        score.backward()
        
        # Grad-CAM++权重
        grads_power_2 = self.gradients ** 2
        grads_power_3 = grads_power_2 * self.gradients
        
        sum_activations = self.activations.sum(dim=(2, 3), keepdim=True)
        
        alpha_numerators = grads_power_2
        alpha_denominators = 2 * grads_power_2 + sum_activations * grads_power_3 + 1e-8
        alphas = alpha_numerators / alpha_denominators
        alphas = alphas.mean(dim=(2, 3), keepdim=True)
        
        weights = (alphas * F.relu(self.gradients)).sum(dim=(2, 3), keepdim=True)
        
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam / (cam.max() + 1e-8)
        
        return cam


# ==================== 注意力可视化 ====================

class AttentionVisualizer:
    """注意力可视化"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self._hooks = []
    
    def register_hooks(self):
        """注册注意力钩子"""
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: self._attention_hook(n, m, i, o)
                )
                self._hooks.append(hook)
    
    def _attention_hook(self, name: str, module: nn.Module, input, output):
        """注意力钩子"""
        if hasattr(module, 'attention_weights'):
            self.attention_maps[name] = module.attention_weights.detach()
    
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """获取注意力图"""
        return self.attention_maps
    
    def clear(self):
        """清除注意力图"""
        self.attention_maps.clear()
    
    def remove_hooks(self):
        """移除钩子"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class AttentionRollout:
    """注意力Rollout"""
    
    def __init__(
        self,
        model: nn.Module,
        head_fusion: str = 'mean',  # 'mean', 'max'
        discard_ratio: float = 0.0,
    ):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attention_maps: List[torch.Tensor] = []
        self._hooks = []
    
    def _attention_hook(self, module, input, output):
        """注意力钩子"""
        if hasattr(module, 'attention_weights'):
            attn = module.attention_weights.detach()
            self.attention_maps.append(attn)
    
    def compute_rollout(self) -> torch.Tensor:
        """计算Rollout"""
        if not self.attention_maps:
            return None
        
        result = torch.eye(self.attention_maps[0].size(-1))
        
        for attn in self.attention_maps:
            # 融合多头
            if self.head_fusion == 'mean':
                attn = attn.mean(dim=1)
            elif self.head_fusion == 'max':
                attn = attn.max(dim=1)[0]
            
            # 添加残差连接
            attn = attn + torch.eye(attn.size(-1), device=attn.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(attn, result)
        
        # 归一化
        mask = result / result.sum(dim=-1, keepdim=True)
        
        return mask


# ==================== 特征归因 ====================

class SHAPExplainer:
    """SHAP解释器（简化实现）"""
    
    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        num_samples: int = 100,
    ):
        self.model = model
        self.background_data = background_data
        self.num_samples = num_samples
        self.model.eval()
    
    @torch.no_grad()
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """计算SHAP值"""
        batch_size = input_tensor.size(0)
        num_features = input_tensor.numel() // batch_size
        
        shap_values = torch.zeros_like(input_tensor)
        
        # 采样背景数据
        indices = torch.randint(0, len(self.background_data), (self.num_samples,))
        backgrounds = self.background_data[indices]
        
        for i in range(self.num_samples):
            # 随机掩码
            mask = torch.rand_like(input_tensor) > 0.5
            
            # 插值
            interpolated = mask * input_tensor + (~mask) * backgrounds[i:i+1]
            
            # 预测
            output = self.model(interpolated)
            
            if target_class is None:
                target_class = output.argmax(dim=-1)
            
            if output.dim() > 1:
                score = output[:, target_class]
            else:
                score = output[target_class]
            
            # 累积
            shap_values += mask.float() * score.unsqueeze(-1)
        
        shap_values = shap_values / self.num_samples
        
        return shap_values


class LIMEExplainer:
    """LIME解释器"""
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 1000,
        num_features: int = 10,
    ):
        self.model = model
        self.num_samples = num_samples
        self.num_features = num_features
        self.model.eval()
    
    @torch.no_grad()
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算LIME解释"""
        # 生成扰动样本
        batch_size = input_tensor.size(0)
        num_superpixels = 50  # 简化：假设50个超像素
        
        # 随机掩码
        masks = torch.randint(0, 2, (self.num_samples, num_superpixels))
        
        # 计算距离权重
        distances = torch.zeros(self.num_samples)
        predictions = []
        
        for i in range(self.num_samples):
            # 应用掩码（简化）
            masked_input = input_tensor * masks[i].float().mean()
            
            output = self.model(masked_input)
            
            if target_class is None:
                target_class = output.argmax(dim=-1)
            
            if output.dim() > 1:
                pred = output[:, target_class].item()
            else:
                pred = output[target_class].item()
            
            predictions.append(pred)
            
            # 计算距离
            distances[i] = torch.norm(masks[i].float() - 0.5)
        
        # 指数核权重
        kernel_width = np.sqrt(num_superpixels) * 0.25
        weights = torch.exp(-distances ** 2 / kernel_width ** 2)
        
        # 加权线性回归
        predictions = torch.tensor(predictions)
        
        # 简化：使用最小二乘
        X = masks.float()
        W = torch.diag(weights)
        
        # (X^T W X)^-1 X^T W y
        XWX = X.t() @ W @ X
        XWy = X.t() @ W @ predictions
        
        try:
            coefficients = torch.linalg.solve(XWX, XWy)
        except:
            coefficients = torch.zeros(num_superpixels)
        
        return coefficients.numpy(), masks.numpy()


# ==================== 概念激活向量 ====================

class ConceptActivationVector:
    """概念激活向量 (CAV)"""
    
    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
    ):
        self.model = model
        self.layer = layer
        
        self.activations = []
        self._hook = None
    
    def _forward_hook(self, module, input, output):
        self.activations.append(output.detach())
    
    def register_hook(self):
        """注册钩子"""
        self._hook = self.layer.register_forward_hook(self._forward_hook)
    
    def remove_hook(self):
        """移除钩子"""
        if self._hook:
            self._hook.remove()
    
    @torch.no_grad()
    def compute_cav(
        self,
        positive_examples: torch.Tensor,
        negative_examples: torch.Tensor,
    ) -> torch.Tensor:
        """计算CAV"""
        self.activations.clear()
        self.register_hook()
        
        # 获取正例激活
        self.model(positive_examples)
        positive_activations = torch.cat(self.activations, dim=0)
        self.activations.clear()
        
        # 获取负例激活
        self.model(negative_examples)
        negative_activations = torch.cat(self.activations, dim=0)
        self.activations.clear()
        
        self.remove_hook()
        
        # 计算CAV（线性分类器法向量）
        pos_mean = positive_activations.mean(dim=0)
        neg_mean = negative_activations.mean(dim=0)
        
        cav = pos_mean - neg_mean
        cav = cav / cav.norm()
        
        return cav
    
    @torch.no_grad()
    def sensitivity(
        self,
        input_tensor: torch.Tensor,
        cav: torch.Tensor,
        target_class: int,
    ) -> float:
        """计算概念敏感性"""
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        self.activations.clear()
        self.register_hook()
        
        output = self.model(input_tensor)
        
        self.remove_hook()
        
        if output.dim() > 1:
            score = output[:, target_class].sum()
        else:
            score = output[target_class].sum()
        
        score.backward()
        
        # 计算方向导数
        activation = self.activations[0]
        grad = input_tensor.grad
        
        sensitivity = torch.sum(grad * cav.flatten()[:grad.numel()].reshape_as(grad))
        
        return sensitivity.item()


# ==================== 反事实解释 ====================

class CounterfactualExplainer:
    """反事实解释器"""
    
    def __init__(
        self,
        model: nn.Module,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
    ):
        self.model = model
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.model.eval()
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        desired_class: int,
        constraints: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, float]:
        """生成反事实解释"""
        # 初始化
        counterfactual = input_tensor.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([counterfactual], lr=self.learning_rate)
        
        for _ in range(self.num_iterations):
            optimizer.zero_grad()
            
            output = self.model(counterfactual)
            
            # 目标：改变预测到desired_class
            loss_pred = -F.log_softmax(output, dim=-1)[:, desired_class].mean()
            
            # 约束：保持接近原始输入
            loss_proximity = F.mse_loss(counterfactual, input_tensor)
            
            # 约束
            loss_constraints = 0.0
            if constraints:
                for key, value in constraints.items():
                    if key == 'l1_norm':
                        loss_constraints += value * torch.norm(counterfactual - input_tensor, p=1)
                    elif key == 'l2_norm':
                        loss_constraints += value * torch.norm(counterfactual - input_tensor, p=2)
            
            total_loss = loss_pred + 0.5 * loss_proximity + loss_constraints
            
            total_loss.backward()
            optimizer.step()
        
        # 计算距离
        distance = F.mse_loss(counterfactual, input_tensor).item()
        
        return counterfactual.detach(), distance


# ==================== 特征重要性 ====================

class PermutationImportance:
    """置换重要性"""
    
    def __init__(
        self,
        model: nn.Module,
        num_repeats: int = 10,
    ):
        self.model = model
        self.num_repeats = num_repeats
        self.model.eval()
    
    @torch.no_grad()
    def compute(
        self,
        input_tensor: torch.Tensor,
        target: torch.Tensor,
        metric: Callable = lambda p, t: (p.argmax(dim=-1) == t).float().mean(),
    ) -> np.ndarray:
        """计算置换重要性"""
        # 基准分数
        output = self.model(input_tensor)
        baseline_score = metric(output, target).item()
        
        num_features = input_tensor.size(-1)
        importances = np.zeros(num_features)
        
        for feature_idx in range(num_features):
            scores = []
            
            for _ in range(self.num_repeats):
                # 置换特征
                permuted = input_tensor.clone()
                perm_indices = torch.randperm(input_tensor.size(0))
                permuted[:, feature_idx] = input_tensor[perm_indices, feature_idx]
                
                output = self.model(permuted)
                score = metric(output, target).item()
                scores.append(score)
            
            # 重要性 = 基准分数 - 置换后分数
            importances[feature_idx] = baseline_score - np.mean(scores)
        
        return importances


class FeatureAblation:
    """特征消融"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def compute(
        self,
        input_tensor: torch.Tensor,
        baseline_value: float = 0.0,
    ) -> np.ndarray:
        """计算特征消融重要性"""
        # 基准输出
        output = self.model(input_tensor)
        baseline_output = output.clone()
        
        num_features = input_tensor.size(-1)
        importances = np.zeros(num_features)
        
        for feature_idx in range(num_features):
            # 消融特征
            ablated = input_tensor.clone()
            ablated[:, feature_idx] = baseline_value
            
            output = self.model(ablated)
            
            # 重要性 = 输出变化
            importances[feature_idx] = (baseline_output - output).abs().mean().item()
        
        return importances


# ==================== 主函数 ====================

def main():
    """测试模型解释性"""
    print("模型解释性测试")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    ).to(device)
    
    # 测试显著性图
    print("\n测试显著性图...")
    saliency = SaliencyMap(model)
    x = torch.randn(1, 3, 32, 32).to(device)
    sal_map = saliency.compute(x, target_class=0)
    print(f"Saliency map shape: {sal_map.shape}")
    
    # 测试积分梯度
    print("\n测试积分梯度...")
    ig = IntegratedGradients(model, num_steps=20)
    ig_map = ig.compute(x, target_class=0)
    print(f"Integrated gradients shape: {ig_map.shape}")
    
    # 测试Grad-CAM
    print("\n测试Grad-CAM...")
    target_layer = model[2]  # 第二个卷积层
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.compute(x, target_class=0)
    print(f"Grad-CAM shape: {cam.shape}")
    
    # 测试置换重要性
    print("\n测试置换重要性...")
    x_flat = torch.randn(10, 32).to(device)
    linear_model = nn.Linear(32, 10).to(device)
    perm_imp = PermutationImportance(linear_model)
    target = torch.randint(0, 10, (10,)).to(device)
    importances = perm_imp.compute(x_flat, target)
    print(f"Permutation importances: {importances[:5]}")
    
    print("\n模型解释性测试完成")


if __name__ == "__main__":
    main()
