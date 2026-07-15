"""
TestLeJEPA - 核心算法单元测试：LeJEPA（联合嵌入预测架构）模块

模块路径: testing/unit/core/test_lejepa.py
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


@dataclass
class PatchConfig:
    patch_size: int = 16
    stride: int = 16
    image_size: int = 224


class MockLeJEPA:
    """模拟LeJEPA模型"""

    def __init__(self, d_model: int = 128, patch_size: int = 16,
                 n_patches: int = 196, n_predictor_layers: int = 4):
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.n_predictor_layers = n_predictor_layers
        self.encoder_weight = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.predictor_weight = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

    def extract_patches(self, image: np.ndarray,
                        patch_size: int = 16, stride: int = 16) -> np.ndarray:
        h, w = image.shape[:2]
        patches = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
        return np.array(patches, dtype=np.float32)

    def patch_to_embedding(self, patches: np.ndarray) -> np.ndarray:
        if patches.ndim == 1:
            patches = patches[np.newaxis, :]
        proj = np.random.randn(patches.shape[-1], self.d_model).astype(np.float32) * 0.02
        return np.matmul(patches, proj)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.encoder_weight)

    def predict(self, context: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
        predicted = np.matmul(context, self.predictor_weight)
        return predicted[target_positions]

    def contrastive_loss(self, z_pred: np.ndarray, z_target: np.ndarray,
                         temperature: float = 0.1) -> float:
        z_pred_norm = z_pred / (np.linalg.norm(z_pred, axis=-1, keepdims=True) + 1e-8)
        z_target_norm = z_target / (np.linalg.norm(z_target, axis=-1, keepdims=True) + 1e-8)
        similarity = np.sum(z_pred_norm * z_target_norm, axis=-1) / temperature
        labels = np.arange(similarity.shape[0])
        exp_sim = np.exp(similarity)
        log_sum_exp = np.log(np.sum(exp_sim, axis=-1) + 1e-8)
        loss = -similarity[np.arange(len(labels)), labels] + log_sum_exp
        return float(np.mean(loss))

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        return np.sum(a_norm * b_norm, axis=-1)

    def mask_patches(self, n_patches: int, mask_ratio: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        n_masked = int(n_patches * mask_ratio)
        n_visible = n_patches - n_masked
        indices = np.random.permutation(n_patches)
        visible = np.sort(indices[:n_visible])
        masked = np.sort(indices[n_visible:])
        return visible, masked

    def compute_repr_similarity(self, repr1: np.ndarray, repr2: np.ndarray) -> Dict[str, float]:
        cos_sim = self.cosine_similarity(repr1, repr2)
        return {
            "mean_cosine_similarity": float(np.mean(cos_sim)),
            "min_cosine_similarity": float(np.min(cos_sim)),
            "max_cosine_similarity": float(np.max(cos_sim)),
            "std_cosine_similarity": float(np.std(cos_sim)),
        }


class TestPatchExtraction:
    """图像块提取测试"""

    def setup_method(self):
        self.lejepa = MockLeJEPA()

    def test_extract_patches_shape(self):
        image = np.random.randn(32, 32, 3).astype(np.float32)
        patches = self.lejepa.extract_patches(image, patch_size=16, stride=16)
        assert patches.ndim == 2
        assert patches.shape[1] == 16 * 16 * 3

    def test_extract_patches_count(self):
        image = np.random.randn(32, 32, 3).astype(np.float32)
        patches = self.lejepa.extract_patches(image, patch_size=16, stride=16)
        assert patches.shape[0] == 4

    def test_extract_patches_stride_8(self):
        image = np.random.randn(32, 32, 1).astype(np.float32)
        patches = self.lejepa.extract_patches(image, patch_size=8, stride=8)
        assert patches.shape[0] == 16

    def test_patch_to_embedding_shape(self):
        patches = np.random.randn(10, 256).astype(np.float32)
        embeddings = self.lejepa.patch_to_embedding(patches)
        assert embeddings.shape == (10, self.lejepa.d_model)


class TestEncoder:
    """编码器测试"""

    def setup_method(self):
        self.lejepa = MockLeJEPA(d_model=64)

    def test_encode_shape(self):
        x = np.random.randn(5, 64).astype(np.float32)
        encoded = self.lejepa.encode(x)
        assert encoded.shape == x.shape

    def test_encode_deterministic(self):
        x = np.random.randn(5, 64).astype(np.float32)
        e1 = self.lejepa.encode(x)
        e2 = self.lejepa.encode(x)
        assert np.allclose(e1, e2)

    def test_encode_different_inputs(self):
        x1 = np.random.randn(5, 64).astype(np.float32)
        x2 = np.random.randn(5, 64).astype(np.float32)
        assert not np.allclose(self.lejepa.encode(x1), self.lejepa.encode(x2))


class TestPredictor:
    """预测器测试"""

    def setup_method(self):
        self.lejepa = MockLeJEPA(d_model=64)

    def test_predict_shape(self):
        context = np.random.randn(10, 64).astype(np.float32)
        target_pos = np.array([0, 2, 4, 6, 8])
        predicted = self.lejepa.predict(context, target_pos)
        assert predicted.shape == (5, 64)

    def test_predict_deterministic(self):
        context = np.random.randn(10, 64).astype(np.float32)
        target_pos = np.array([0, 1, 2])
        p1 = self.lejepa.predict(context, target_pos)
        p2 = self.lejepa.predict(context, target_pos)
        assert np.allclose(p1, p2)


class TestContrastiveLoss:
    """对比损失测试"""

    def setup_method(self):
        self.lejepa = MockLeJEPA(d_model=64)

    def test_loss_positive(self):
        z_pred = np.random.randn(8, 64).astype(np.float32)
        z_target = np.random.randn(8, 64).astype(np.float32)
        loss = self.lejepa.contrastive_loss(z_pred, z_target)
        assert loss >= 0

    def test_loss_identical_embeddings_low(self):
        z = np.random.randn(8, 64).astype(np.float32)
        z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        loss = self.lejepa.contrastive_loss(z_norm, z_norm, temperature=0.01)
        assert loss < 1.0

    def test_loss_temperature_effect(self):
        z_pred = np.random.randn(8, 64).astype(np.float32)
        z_target = np.random.randn(8, 64).astype(np.float32)
        loss_low_t = self.lejepa.contrastive_loss(z_pred, z_target, temperature=0.01)
        loss_high_t = self.lejepa.contrastive_loss(z_pred, z_target, temperature=1.0)
        assert loss_low_t != loss_high_t


class TestPatchMasking:
    """图像块遮蔽测试"""

    def setup_method(self):
        self.lejepa = MockLeJEPA()

    def test_mask_ratio(self):
        visible, masked = self.lejepa.mask_patches(100, mask_ratio=0.75)
        assert len(visible) == 25
        assert len(masked) == 75

    def test_no_overlap(self):
        visible, masked = self.lejepa.mask_patches(100, mask_ratio=0.5)
        assert len(set(visible) & set(masked)) == 0

    def test_complete_coverage(self):
        visible, masked = self.lejepa.mask_patches(50, mask_ratio=0.6)
        assert len(visible) + len(masked) == 50

    def test_mask_ratio_zero(self):
        visible, masked = self.lejepa.mask_patches(100, mask_ratio=0.0)
        assert len(visible) == 100
        assert len(masked) == 0


class TestRepresentationSimilarity:
    """表征相似度测试"""

    def setup_method(self):
        self.lejepa = MockLeJEPA(d_model=64)

    def test_identical_reprs(self):
        r = np.random.randn(5, 64).astype(np.float32)
        stats = self.lejepa.compute_repr_similarity(r, r)
        assert stats["mean_cosine_similarity"] > 0.99

    def test_orthogonal_reprs(self):
        r1 = np.eye(64, dtype=np.float32)[:5]
        r2 = np.eye(64, dtype=np.float32)[5:10]
        stats = self.lejepa.compute_repr_similarity(r1, r2)
        assert stats["mean_cosine_similarity"] < 0.1

    def test_stats_keys(self):
        r1 = np.random.randn(5, 64).astype(np.float32)
        r2 = np.random.randn(5, 64).astype(np.float32)
        stats = self.lejepa.compute_repr_similarity(r1, r2)
        assert "mean_cosine_similarity" in stats
        assert "min_cosine_similarity" in stats
        assert "max_cosine_similarity" in stats
        assert "std_cosine_similarity" in stats
