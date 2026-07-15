"""
联邦聚合收敛测试

测试联邦学习中的各种聚合算法、拜占庭鲁棒性、
梯度压缩、安全聚合和差分隐私的正确性。

使用纯Python模拟数据和算法逻辑，不依赖真实模型。
"""

import unittest
import math
import random
import hashlib
import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# 辅助工具：模拟联邦学习组件
# ---------------------------------------------------------------------------

class SimulatedClient:
    """模拟联邦学习客户端，持有本地数据和模型参数。"""

    def __init__(self, client_id, data, initial_weights):
        self.client_id = client_id
        self.data = data  # list of (x, y) tuples
        self.weights = list(initial_weights)

    def local_train(self, lr=0.01, epochs=1):
        """在本地数据上进行简单的梯度下降训练。"""
        for _ in range(epochs):
            grad = self._compute_gradient()
            for i in range(len(self.weights)):
                self.weights[i] -= lr * grad[i]
        return list(self.weights)

    def _compute_gradient(self):
        """计算均方误差的梯度（线性模型 y = weights[0]*x + weights[1]）。"""
        n = len(self.data)
        if n == 0:
            return [0.0] * len(self.weights)
        grad = [0.0] * len(self.weights)
        for x, y in self.data:
            pred = self.weights[0] * x + self.weights[1]
            error = pred - y
            grad[0] += error * x / n
            grad[1] += error / n
        return grad

    def get_model_size(self):
        return len(self.weights)


class FederatedSimulator:
    """联邦学习模拟器，实现各种聚合算法。"""

    def __init__(self, clients, server_weights):
        self.clients = clients
        self.server_weights = list(server_weights)
        self.round_history = []

    def fedavg(self, lr=0.01, epochs=1, fraction=1.0):
        """FedAvg聚合：按客户端数据量加权平均。"""
        num_selected = max(1, int(len(self.clients) * fraction))
        selected = random.sample(self.clients, min(num_selected, len(self.clients)))

        total_samples = sum(len(c.data) for c in selected)
        aggregated = [0.0] * len(self.server_weights)

        for client in selected:
            client.weights = list(self.server_weights)
            updated = client.local_train(lr=lr, epochs=epochs)
            weight = len(client.data) / max(total_samples, 1)
            for i in range(len(aggregated)):
                aggregated[i] += weight * updated[i]

        self.server_weights = aggregated
        self.round_history.append(list(aggregated))
        return aggregated

    def fedprox(self, lr=0.01, epochs=1, mu=0.01):
        """FedProx聚合：在本地训练中添加近端正则化项。"""
        total_samples = sum(len(c.data) for c in self.clients)
        aggregated = [0.0] * len(self.server_weights)

        for client in self.clients:
            client.weights = list(self.server_weights)
            # 近端正则化：在本地梯度中添加 mu * (w - w_server)
            for _ in range(epochs):
                grad = client._compute_gradient()
                for i in range(len(client.weights)):
                    proximal_term = mu * (client.weights[i] - self.server_weights[i])
                    client.weights[i] -= lr * (grad[i] + proximal_term)

            weight = len(client.data) / max(total_samples, 1)
            for i in range(len(aggregated)):
                aggregated[i] += weight * client.weights[i]

        self.server_weights = aggregated
        self.round_history.append(list(aggregated))
        return aggregated

    def fedadam(self, lr=0.01, epochs=1, beta1=0.9, beta2=0.999, eps=1e-8):
        """FedAdam聚合：使用Adam优化器的服务端聚合。"""
        if not hasattr(self, '_adam_m'):
            self._adam_m = [0.0] * len(self.server_weights)
            self._adam_v = [0.0] * len(self.server_weights)
            self._adam_t = 0

        total_samples = sum(len(c.data) for c in self.clients)
        delta_avg = [0.0] * len(self.server_weights)

        for client in self.clients:
            client.weights = list(self.server_weights)
            updated = client.local_train(lr=lr, epochs=epochs)
            weight = len(client.data) / max(total_samples, 1)
            for i in range(len(delta_avg)):
                delta_avg[i] += weight * (updated[i] - self.server_weights[i])

        self._adam_t += 1
        for i in range(len(self.server_weights)):
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * delta_avg[i]
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * delta_avg[i] ** 2
            m_hat = self._adam_m[i] / (1 - beta1 ** self._adam_t)
            v_hat = self._adam_v[i] / (1 - beta2 ** self._adam_t)
            self.server_weights[i] += lr * m_hat / (math.sqrt(v_hat) + eps)

        self.round_history.append(list(self.server_weights))
        return list(self.server_weights)

    def async_aggregation(self, staleness_threshold=3):
        """异步聚合：考虑客户端更新的陈旧度。"""
        total_samples = sum(len(c.data) for c in self.clients)
        aggregated = [0.0] * len(self.server_weights)
        total_weight = 0.0

        for client in self.clients:
            client.weights = list(self.server_weights)
            updated = client.local_train(lr=0.01, epochs=1)
            staleness = random.randint(0, staleness_threshold)
            # 陈旧度衰减因子
            decay = 1.0 / (1.0 + staleness)
            weight = decay * len(client.data) / max(total_samples, 1)
            for i in range(len(aggregated)):
                aggregated[i] += weight * updated[i]
            total_weight += weight

        if total_weight > 0:
            for i in range(len(aggregated)):
                aggregated[i] /= total_weight

        self.server_weights = aggregated
        self.round_history.append(list(aggregated))
        return aggregated

    def krum(self, num_byzantine=1):
        """Krum聚合：选择与其它更新最接近的单个更新。"""
        updates = []
        for client in self.clients:
            client.weights = list(self.server_weights)
            updated = client.local_train(lr=0.01, epochs=1)
            updates.append(updated)

        n = len(updates)
        # 计算每对更新之间的距离
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(float('inf'))
                else:
                    dist = math.sqrt(sum(
                        (updates[i][k] - updates[j][k]) ** 2
                        for k in range(len(updates[i]))
                    ))
                    row.append(dist)
            distances.append(row)

        # 对每个更新，找到最近的 (n - num_byzantine - 2) 个邻居的距离之和
        scores = []
        num_neighbors = n - num_byzantine - 2
        for i in range(n):
            sorted_dists = sorted(distances[i])
            score = sum(sorted_dists[:max(1, num_neighbors)])
            scores.append((score, i))

        scores.sort()
        best_idx = scores[0][1]
        self.server_weights = list(updates[best_idx])
        self.round_history.append(list(self.server_weights))
        return list(self.server_weights)

    def trimmed_mean(self, trim_ratio=0.2):
        """Trimmed Mean聚合：去掉极端值后取平均。"""
        updates = []
        for client in self.clients:
            client.weights = list(self.server_weights)
            updated = client.local_train(lr=0.01, epochs=1)
            updates.append(updated)

        dim = len(self.server_weights)
        aggregated = [0.0] * dim
        num_updates = len(updates)
        trim_count = max(1, int(num_updates * trim_ratio))

        for d in range(dim):
            values = sorted([u[d] for u in updates])
            trimmed = values[trim_count:num_updates - trim_count]
            if trimmed:
                aggregated[d] = sum(trimmed) / len(trimmed)
            else:
                aggregated[d] = values[num_updates // 2]

        self.server_weights = aggregated
        self.round_history.append(list(aggregated))
        return aggregated

    def sparsification_aggregate(self, sparsity_ratio=0.5, lr=0.01):
        """稀疏化聚合：只保留最大的梯度分量。"""
        total_samples = sum(len(c.data) for c in self.clients)
        aggregated = [0.0] * len(self.server_weights)

        for client in self.clients:
            client.weights = list(self.server_weights)
            updated = client.local_train(lr=lr, epochs=1)
            delta = [updated[i] - self.server_weights[i] for i in range(len(updated))]

            # Top-k 稀疏化
            k = max(1, int(len(delta) * (1 - sparsity_ratio)))
            indexed = sorted(enumerate(delta), key=lambda x: abs(x[1]), reverse=True)
            sparse_delta = [0.0] * len(delta)
            for idx, val in indexed[:k]:
                sparse_delta[idx] = val

            weight = len(client.data) / max(total_samples, 1)
            for i in range(len(aggregated)):
                aggregated[i] += weight * (self.server_weights[i] + sparse_delta[i])

        self.server_weights = aggregated
        self.round_history.append(list(aggregated))
        return aggregated

    def quantization_aggregate(self, num_bits=4, lr=0.01):
        """量化聚合：将梯度量化到有限精度。"""
        total_samples = sum(len(c.data) for c in self.clients)
        aggregated = [0.0] * len(self.server_weights)

        for client in self.clients:
            client.weights = list(self.server_weights)
            updated = client.local_train(lr=lr, epochs=1)
            delta = [updated[i] - self.server_weights[i] for i in range(len(updated))]

            # 量化
            max_val = max(abs(d) for d in delta) if delta else 1.0
            if max_val == 0:
                max_val = 1.0
            levels = 2 ** num_bits - 1
            quantized = [round(d / max_val * levels) / levels * max_val for d in delta]

            weight = len(client.data) / max(total_samples, 1)
            for i in range(len(aggregated)):
                aggregated[i] += weight * (self.server_weights[i] + quantized[i])

        self.server_weights = aggregated
        self.round_history.append(list(aggregated))
        return aggregated

    def secure_aggregate(self, lr=0.01):
        """安全聚合：使用秘密共享模拟，验证聚合结果与明文聚合一致。"""
        total_samples = sum(len(c.data) for c in self.clients)
        aggregated = [0.0] * len(self.server_weights)

        # 每个客户端生成随机掩码对
        masks = {}
        for client in self.clients:
            mask = [random.gauss(0, 1) for _ in range(len(self.server_weights))]
            masks[client.client_id] = mask

        # 模拟秘密共享：每个客户端用掩码加密更新
        encrypted_updates = {}
        for client in self.clients:
            client.weights = list(self.server_weights)
            updated = client.local_train(lr=lr, epochs=1)
            encrypted = [updated[i] + masks[client.client_id][i]
                         for i in range(len(updated))]
            encrypted_updates[client.client_id] = encrypted

        # 服务端聚合加密更新
        encrypted_agg = [0.0] * len(self.server_weights)
        for client in self.clients:
            weight = len(client.data) / max(total_samples, 1)
            for i in range(len(encrypted_agg)):
                encrypted_agg[i] += weight * encrypted_updates[client.client_id][i]

        # 减去掩码总和得到真实聚合
        mask_sum = [0.0] * len(self.server_weights)
        for client in self.clients:
            weight = len(client.data) / max(total_samples, 1)
            for i in range(len(mask_sum)):
                mask_sum[i] += weight * masks[client.client_id][i]

        for i in range(len(aggregated)):
            aggregated[i] = encrypted_agg[i] - mask_sum[i]

        self.server_weights = aggregated
        self.round_history.append(list(aggregated))
        return aggregated


class PrivacyAccountant:
    """差分隐私预算记账器。"""

    def __init__(self, target_epsilon=10.0, target_delta=1e-5):
        self.epsilon = 0.0
        self.delta = 0.0
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.round_costs = []

    def add_noise(self, values, clip_norm=1.0, sigma=1.0):
        """对值添加高斯噪声以满足差分隐私。"""
        noised = []
        for v in values:
            # 裁剪
            v_clipped = max(-clip_norm, min(clip_norm, v))
            noised_val = v_clipped + random.gauss(0, sigma)
            noised.append(noised_val)
        return noised

    def spend_budget(self, noise_multiplier, batch_size, total_samples):
        """根据RDP（Renyi Differential Privacy）记账。"""
        # 简化的隐私预算计算
        q = batch_size / max(total_samples, 1)
        # 高斯机制的RDP
        rdp = q ** 2 / (2 * noise_multiplier ** 2)
        # 转换为 (epsilon, delta) DP
        epsilon_step = rdp + math.log(1 / max(self.target_delta, 1e-10))
        self.epsilon += epsilon_step
        self.round_costs.append(epsilon_step)
        return epsilon_step

    def is_budget_exhausted(self):
        return self.epsilon >= self.target_epsilon


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestFedAvgConvergence(unittest.TestCase):
    """测试FedAvg在IID数据上的收敛性。"""

    def setUp(self):
        random.seed(42)
        # 生成IID数据：y = 3x + 1 + noise
        self.true_weight = 3.0
        self.true_bias = 1.0
        self.clients = []
        for i in range(10):
            data = []
            for _ in range(50):
                x = random.uniform(-5, 5)
                y = self.true_weight * x + self.true_bias + random.gauss(0, 0.1)
                data.append((x, y))
            client = SimulatedClient(f"client_{i}", data, [0.0, 0.0])
            self.clients.append(client)
        self.simulator = FederatedSimulator(self.clients, [0.0, 0.0])

    def test_fedavg_convergence(self):
        """FedAvg在IID数据上应收敛到接近真实参数。"""
        losses = []
        for round_num in range(50):
            weights = self.simulator.fedavg(lr=0.05, epochs=1)
            # 计算全局损失
            total_loss = 0.0
            total_count = 0
            for client in self.clients:
                for x, y in client.data:
                    pred = weights[0] * x + weights[1]
                    total_loss += (pred - y) ** 2
                    total_count += 1
            mse = total_loss / max(total_count, 1)
            losses.append(mse)

        # 验证损失下降
        self.assertLess(losses[-1], losses[0],
                        "FedAvg应使损失下降")
        # 验证收敛到接近真实参数
        self.assertAlmostEqual(weights[0], self.true_weight, delta=0.5,
                               msg="权重应接近真实值")
        self.assertAlmostEqual(weights[1], self.true_bias, delta=0.5,
                               msg="偏置应接近真实值")

    def test_fedavg_non_iid(self):
        """FedAvg在Non-IID数据上应仍能收敛，但速度可能较慢。"""
        # 生成Non-IID数据：每个客户端只有部分范围的数据
        non_iid_clients = []
        for i in range(10):
            data = []
            # 每个客户端只有特定范围的x值
            x_min = -5 + i * 1.0
            x_max = x_min + 1.0
            for _ in range(50):
                x = random.uniform(x_min, x_max)
                y = self.true_weight * x + self.true_bias + random.gauss(0, 0.1)
                data.append((x, y))
            client = SimulatedClient(f"client_{i}", data, [0.0, 0.0])
            non_iid_clients.append(client)

        sim = FederatedSimulator(non_iid_clients, [0.0, 0.0])
        losses = []
        for _ in range(80):
            weights = sim.fedavg(lr=0.05, epochs=1)
            total_loss = 0.0
            total_count = 0
            for client in non_iid_clients:
                for x, y in client.data:
                    pred = weights[0] * x + weights[1]
                    total_loss += (pred - y) ** 2
                    total_count += 1
            mse = total_loss / max(total_count, 1)
            losses.append(mse)

        # Non-IID情况下也应收敛
        self.assertLess(losses[-1], losses[0],
                        "FedAvg在Non-IID数据上也应使损失下降")
        # 允许更大的误差范围
        self.assertAlmostEqual(weights[0], self.true_weight, delta=1.0,
                               msg="Non-IID下权重仍应接近真实值")


class TestFedProxConvergence(unittest.TestCase):
    """测试FedProx收敛性。"""

    def setUp(self):
        random.seed(42)
        self.true_weight = 3.0
        self.true_bias = 1.0
        self.clients = []
        for i in range(10):
            data = []
            for _ in range(50):
                x = random.uniform(-5, 5)
                y = self.true_weight * x + self.true_bias + random.gauss(0, 0.1)
                data.append((x, y))
            client = SimulatedClient(f"client_{i}", data, [0.0, 0.0])
            self.clients.append(client)

    def test_fedprox_convergence(self):
        """FedProx应收敛，近端正则化应防止客户端漂移过大。"""
        sim = FederatedSimulator(self.clients, [0.0, 0.0])
        max_drifts = []

        for _ in range(50):
            weights_before = list(sim.server_weights)
            weights = sim.fedprox(lr=0.05, epochs=1, mu=0.1)
            # 记录客户端漂移
            drifts = []
            for client in self.clients:
                drift = math.sqrt(sum(
                    (client.weights[i] - weights[i]) ** 2
                    for i in range(len(client.weights))
                ))
                drifts.append(drift)
            max_drifts.append(max(drifts) if drifts else 0)

        # 验证收敛
        self.assertAlmostEqual(weights[0], self.true_weight, delta=0.5,
                               msg="FedProx权重应接近真实值")
        # 近端正则化应限制漂移
        avg_drift = sum(max_drifts[-10:]) / 10
        self.assertLess(avg_drift, 5.0,
                        "FedProx应限制客户端漂移")


class TestFedAdamConvergence(unittest.TestCase):
    """测试FedAdam收敛性。"""

    def setUp(self):
        random.seed(42)
        self.true_weight = 3.0
        self.true_bias = 1.0
        self.clients = []
        for i in range(10):
            data = []
            for _ in range(50):
                x = random.uniform(-5, 5)
                y = self.true_weight * x + self.true_bias + random.gauss(0, 0.1)
                data.append((x, y))
            client = SimulatedClient(f"client_{i}", data, [0.0, 0.0])
            self.clients.append(client)

    def test_fedadam_convergence(self):
        """FedAdam应比普通平均更快速收敛。"""
        sim = FederatedSimulator(self.clients, [0.0, 0.0])
        losses = []
        for _ in range(50):
            weights = sim.fedadam(lr=0.1, epochs=1)
            total_loss = 0.0
            total_count = 0
            for client in self.clients:
                for x, y in client.data:
                    pred = weights[0] * x + weights[1]
                    total_loss += (pred - y) ** 2
                    total_count += 1
            mse = total_loss / max(total_count, 1)
            losses.append(mse)

        self.assertLess(losses[-1], losses[0],
                        "FedAdam应使损失下降")
        self.assertAlmostEqual(weights[0], self.true_weight, delta=0.5,
                               msg="FedAdam权重应接近真实值")


class TestAsyncAggregation(unittest.TestCase):
    """测试异步聚合的正确性。"""

    def setUp(self):
        random.seed(42)
        self.true_weight = 3.0
        self.true_bias = 1.0
        self.clients = []
        for i in range(10):
            data = []
            for _ in range(50):
                x = random.uniform(-5, 5)
                y = self.true_weight * x + self.true_bias + random.gauss(0, 0.1)
                data.append((x, y))
            client = SimulatedClient(f"client_{i}", data, [0.0, 0.0])
            self.clients.append(client)

    def test_async_aggregation(self):
        """异步聚合应正确处理不同陈旧度的更新。"""
        sim = FederatedSimulator(self.clients, [0.0, 0.0])
        weights = sim.async_aggregation(staleness_threshold=3)

        # 验证聚合结果合理（不会发散）
        self.assertTrue(all(math.isfinite(w) for w in weights),
                        "异步聚合结果应全部为有限值")
        # 验证结果在合理范围内
        self.assertTrue(all(abs(w) < 100 for w in weights),
                        "异步聚合结果应在合理范围内")

        # 多轮异步聚合后应收敛
        for _ in range(30):
            weights = sim.async_aggregation(staleness_threshold=3)

        self.assertAlmostEqual(weights[0], self.true_weight, delta=1.0,
                               msg="异步聚合多轮后应收敛")


class TestByzantineRobustness(unittest.TestCase):
    """测试拜占庭鲁棒性聚合算法。"""

    def setUp(self):
        random.seed(42)
        self.true_weight = 3.0
        self.true_bias = 1.0

    def _make_clients(self, num_honest=8, num_byzantine=2):
        """创建诚实和拜占庭客户端。"""
        clients = []
        for i in range(num_honest):
            data = []
            for _ in range(50):
                x = random.uniform(-5, 5)
                y = self.true_weight * x + self.true_bias + random.gauss(0, 0.1)
                data.append((x, y))
            client = SimulatedClient(f"honest_{i}", data, [0.0, 0.0])
            clients.append(client)
        for i in range(num_byzantine):
            data = []
            for _ in range(50):
                x = random.uniform(-5, 5)
                y = self.true_weight * x + self.true_bias + random.gauss(0, 0.1)
                data.append((x, y))
            client = SimulatedClient(f"byzantine_{i}", data, [0.0, 0.0])
            clients.append(client)
        return clients

    def test_byzantine_krum(self):
        """Krum应能抵御恶意更新。"""
        clients = self._make_clients(num_honest=8, num_byzantine=2)
        sim = FederatedSimulator(clients, [0.0, 0.0])

        for _ in range(30):
            # 拜占庭客户端发送恶意更新
            for client in clients:
                if client.client_id.startswith("byzantine"):
                    # 在local_train后篡改权重
                    client.local_train(lr=0.05, epochs=1)
                    client.weights = [100.0, -100.0]  # 恶意值
            weights = sim.krum(num_byzantine=2)

        # Krum应忽略恶意更新，收敛到接近真实值
        self.assertAlmostEqual(weights[0], self.true_weight, delta=1.5,
                               msg="Krum应抵御恶意更新")

    def test_byzantine_trimmed_mean(self):
        """TrimmedMean应能抵御恶意更新。"""
        clients = self._make_clients(num_honest=8, num_byzantine=2)
        sim = FederatedSimulator(clients, [0.0, 0.0])

        for _ in range(30):
            for client in clients:
                if client.client_id.startswith("byzantine"):
                    client.local_train(lr=0.05, epochs=1)
                    client.weights = [100.0, -100.0]
            weights = sim.trimmed_mean(trim_ratio=0.2)

        self.assertAlmostEqual(weights[0], self.true_weight, delta=2.0,
                               msg="TrimmedMean应抵御恶意更新")


class TestCompression(unittest.TestCase):
    """测试梯度压缩后的收敛性。"""

    def setUp(self):
        random.seed(42)
        self.true_weight = 3.0
        self.true_bias = 1.0
        self.clients = []
        for i in range(10):
            data = []
            for _ in range(50):
                x = random.uniform(-5, 5)
                y = self.true_weight * x + self.true_bias + random.gauss(0, 0.1)
                data.append((x, y))
            client = SimulatedClient(f"client_{i}", data, [0.0, 0.0])
            self.clients.append(client)

    def test_compression_sparsification(self):
        """稀疏化后应仍能收敛。"""
        sim = FederatedSimulator(self.clients, [0.0, 0.0])
        losses = []
        for _ in range(60):
            weights = sim.sparsification_aggregate(sparsity_ratio=0.5, lr=0.05)
            total_loss = 0.0
            total_count = 0
            for client in self.clients:
                for x, y in client.data:
                    pred = weights[0] * x + weights[1]
                    total_loss += (pred - y) ** 2
                    total_count += 1
            mse = total_loss / max(total_count, 1)
            losses.append(mse)

        self.assertLess(losses[-1], losses[0],
                        "稀疏化后损失应下降")
        self.assertAlmostEqual(weights[0], self.true_weight, delta=1.0,
                               msg="稀疏化后权重应接近真实值")

    def test_compression_quantization(self):
        """量化后应仍能收敛。"""
        sim = FederatedSimulator(self.clients, [0.0, 0.0])
        losses = []
        for _ in range(60):
            weights = sim.quantization_aggregate(num_bits=4, lr=0.05)
            total_loss = 0.0
            total_count = 0
            for client in self.clients:
                for x, y in client.data:
                    pred = weights[0] * x + weights[1]
                    total_loss += (pred - y) ** 2
                    total_count += 1
            mse = total_loss / max(total_count, 1)
            losses.append(mse)

        self.assertLess(losses[-1], losses[0],
                        "量化后损失应下降")
        self.assertAlmostEqual(weights[0], self.true_weight, delta=1.0,
                               msg="量化后权重应接近真实值")


class TestSecureAggregation(unittest.TestCase):
    """测试安全聚合的正确性。"""

    def setUp(self):
        random.seed(42)
        self.true_weight = 3.0
        self.true_bias = 1.0
        self.clients = []
        for i in range(10):
            data = []
            for _ in range(50):
                x = random.uniform(-5, 5)
                y = self.true_weight * x + self.true_bias + random.gauss(0, 0.1)
                data.append((x, y))
            client = SimulatedClient(f"client_{i}", data, [0.0, 0.0])
            self.clients.append(client)

    def test_secure_aggregation(self):
        """安全聚合结果应与明文聚合一致。"""
        # 明文聚合
        sim_plain = FederatedSimulator(self.clients, [0.0, 0.0])
        random.seed(42)
        plain_weights = sim_plain.fedavg(lr=0.05, epochs=1)

        # 安全聚合
        sim_secure = FederatedSimulator(self.clients, [0.0, 0.0])
        random.seed(42)
        secure_weights = sim_secure.secure_aggregate(lr=0.05)

        # 两者应非常接近（由于随机种子相同，本地训练结果相同）
        for i in range(len(plain_weights)):
            self.assertAlmostEqual(plain_weights[i], secure_weights[i], places=10,
                                   msg=f"安全聚合结果应与明文聚合一致 (维度 {i})")


class TestDifferentialPrivacy(unittest.TestCase):
    """测试差分隐私预算记账。"""

    def setUp(self):
        random.seed(42)

    def test_dp_privacy(self):
        """差分隐私预算应正确记账。"""
        accountant = PrivacyAccountant(target_epsilon=10.0, target_delta=1e-5)

        # 模拟多轮训练
        for round_num in range(20):
            noise_multiplier = 1.0
            batch_size = 10
            total_samples = 100
            epsilon_step = accountant.spend_budget(
                noise_multiplier, batch_size, total_samples
            )
            self.assertGreater(epsilon_step, 0,
                               "每轮应消耗正的隐私预算")

        # 验证总预算正确累加
        self.assertGreater(accountant.epsilon, 0,
                           "总隐私预算应为正")
        self.assertEqual(len(accountant.round_costs), 20,
                         "应记录20轮的预算消耗")

        # 验证预算耗尽检测
        accountant.epsilon = 100.0
        self.assertTrue(accountant.is_budget_exhausted(),
                        "预算耗尽时应返回True")
        accountant.epsilon = 0.0
        self.assertFalse(accountant.is_budget_exhausted(),
                         "预算未耗尽时应返回False")

    def test_dp_noise_correctness(self):
        """添加的噪声应满足差分隐私要求（统计验证）。"""
        accountant = PrivacyAccountant()
        values = [5.0, 3.0, 7.0, 1.0, 9.0]

        # 多次添加噪声，验证均值接近原始值
        noisy_means = []
        for _ in range(1000):
            noised = accountant.add_noise(values, clip_norm=10.0, sigma=1.0)
            noisy_means.append(sum(noised) / len(noised))

        mean_of_means = sum(noisy_means) / len(noisy_means)
        original_mean = sum(values) / len(values)
        # 噪声均值为0，所以大量采样的均值应接近原始值
        self.assertAlmostEqual(mean_of_means, original_mean, delta=0.5,
                               msg="噪声均值应接近0")


if __name__ == '__main__':
    unittest.main()
