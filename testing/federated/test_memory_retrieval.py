"""
加密记忆检索测试

测试联邦学习框架中的加密存储、向量搜索、同步调度、
对抗样本处理、知识蒸馏、知识冲突解决、信誉系统和贡献证明。

使用纯Python标准库实现，不依赖外部加密库。
"""

import unittest
import math
import random
import hashlib
import hmac
import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# 辅助工具：模拟加密记忆系统组件
# ---------------------------------------------------------------------------

class SimpleEncryptor:
    """基于XOR的简单加密器（仅用于测试，非生产级加密）。"""

    def __init__(self, key="default_key_for_testing"):
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, plaintext):
        """加密明文数据。"""
        if isinstance(plaintext, str):
            data = plaintext.encode('utf-8')
        else:
            data = plaintext
        key_stream = (self.key * (len(data) // len(self.key) + 1))[:len(data)]
        encrypted = bytes(a ^ b for a, b in zip(data, key_stream))
        return encrypted

    def decrypt(self, ciphertext):
        """解密密文数据。"""
        key_stream = (self.key * (len(ciphertext) // len(self.key) + 1))[:len(ciphertext)]
        decrypted = bytes(a ^ b for a, b in zip(ciphertext, key_stream))
        return decrypted


class EncryptedMemoryStore:
    """加密记忆存储系统。"""

    def __init__(self, encryptor=None):
        self.encryptor = encryptor or SimpleEncryptor()
        self.store = {}  # key -> encrypted_value
        self.metadata = {}  # key -> metadata dict

    def store_memory(self, key, value, metadata=None):
        """加密并存储记忆。"""
        if isinstance(value, str):
            encrypted = self.encryptor.encrypt(value)
        else:
            encrypted = self.encryptor.encrypt(str(value).encode('utf-8'))
        self.store[key] = encrypted
        self.metadata[key] = metadata or {}

    def retrieve_memory(self, key):
        """检索并解密记忆。"""
        if key not in self.store:
            return None
        encrypted = self.store[key]
        decrypted = self.encryptor.decrypt(encrypted)
        try:
            return decrypted.decode('utf-8')
        except UnicodeDecodeError:
            return decrypted

    def has_memory(self, key):
        """检查记忆是否存在。"""
        return key in self.store

    def delete_memory(self, key):
        """删除记忆。"""
        if key in self.store:
            del self.store[key]
            del self.metadata[key]
            return True
        return False


class VectorIndex:
    """简单的向量索引，支持加密向量的近邻搜索。"""

    def __init__(self, encryptor=None):
        self.encryptor = encryptor or SimpleEncryptor("vector_key")
        self.vectors = {}  # id -> vector (plaintext, 仅在内存中)
        self.encrypted_vectors = {}  # id -> encrypted vector

    def add_vector(self, vec_id, vector):
        """添加向量到索引。"""
        self.vectors[vec_id] = list(vector)
        # 加密存储
        vec_str = ','.join(map(str, vector))
        self.encrypted_vectors[vec_id] = self.encryptor.encrypt(vec_str)

    def _cosine_similarity(self, a, b):
        """计算余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query_vector, top_k=5):
        """搜索最相似的向量。"""
        similarities = []
        for vec_id, vector in self.vectors.items():
            sim = self._cosine_similarity(query_vector, vector)
            similarities.append((sim, vec_id))
        similarities.sort(reverse=True)
        return [(vec_id, sim) for sim, vec_id in similarities[:top_k]]

    def get_encrypted_vector(self, vec_id):
        """获取加密后的向量。"""
        return self.encrypted_vectors.get(vec_id)

    def decrypt_vector(self, vec_id):
        """解密向量。"""
        if vec_id not in self.encrypted_vectors:
            return None
        decrypted = self.encryptor.decrypt(self.encrypted_vectors[vec_id])
        vec_str = decrypted.decode('utf-8')
        return [float(x) for x in vec_str.split(',')]


class SyncScheduler:
    """联邦记忆同步调度器。"""

    def __init__(self, num_nodes=5):
        self.num_nodes = num_nodes
        self.node_versions = {i: 0 for i in range(num_nodes)}
        self.sync_log = []
        self.conflict_resolution_count = 0

    def local_update(self, node_id):
        """模拟节点本地更新。"""
        self.node_versions[node_id] += 1
        self.sync_log.append(('local', node_id, self.node_versions[node_id]))

    def sync_round(self):
        """执行一轮同步。"""
        max_version = max(self.node_versions.values())
        min_version = min(self.node_versions.values())
        if max_version != min_version:
            self.conflict_resolution_count += 1
            # 将所有节点同步到最高版本
            for node_id in self.node_versions:
                self.node_versions[node_id] = max_version
            self.sync_log.append(('sync', 'all', max_version))
            return True
        return False

    def is_consistent(self):
        """检查所有节点是否一致。"""
        versions = set(self.node_versions.values())
        return len(versions) == 1

    def get_sync_log(self):
        """获取同步日志。"""
        return list(self.sync_log)


class AdversarialMemory:
    """对抗样本存储和检索系统。"""

    def __init__(self):
        self.samples = []
        self.perturbation_budget = 0.1

    def store_adversarial(self, original, adversarial, label, attack_type):
        """存储对抗样本。"""
        self.samples.append({
            'original': list(original),
            'adversarial': list(adversarial),
            'label': label,
            'attack_type': attack_type,
            'perturbation': self._compute_perturbation(original, adversarial)
        })

    def _compute_perturbation(self, original, adversarial):
        """计算扰动大小。"""
        return math.sqrt(sum(
            (o - a) ** 2 for o, a in zip(original, adversarial)
        ))

    def retrieve_by_attack_type(self, attack_type):
        """按攻击类型检索对抗样本。"""
        return [s for s in self.samples if s['attack_type'] == attack_type]

    def retrieve_within_budget(self, budget=None):
        """检索扰动在预算内的对抗样本。"""
        budget = budget or self.perturbation_budget
        return [s for s in self.samples if s['perturbation'] <= budget]

    def validate_perturbation(self, original, adversarial, budget=None):
        """验证扰动是否在预算内。"""
        budget = budget or self.perturbation_budget
        pert = self._compute_perturbation(original, adversarial)
        return pert <= budget


class KnowledgeDistiller:
    """联邦知识蒸馏器。"""

    def __init__(self, num_participants=3):
        self.participants = {}
        self.global_knowledge = {}
        self.distillation_history = []

    def add_participant(self, node_id, local_knowledge):
        """添加参与者的本地知识。"""
        self.participants[node_id] = dict(local_knowledge)

    def distill(self, aggregation='weighted_average'):
        """执行知识蒸馏。"""
        if not self.participants:
            return {}

        all_keys = set()
        for knowledge in self.participants.values():
            all_keys.update(knowledge.keys())

        distilled = {}
        for key in all_keys:
            values = []
            for node_id, knowledge in self.participants.items():
                if key in knowledge:
                    values.append(knowledge[key])

            if aggregation == 'weighted_average':
                distilled[key] = sum(values) / len(values) if values else 0
            elif aggregation == 'max':
                distilled[key] = max(values) if values else 0
            elif aggregation == 'min':
                distilled[key] = min(values) if values else 0

        self.global_knowledge = distilled
        self.distillation_history.append(dict(distilled))
        return distilled

    def get_knowledge_conflicts(self):
        """检测知识冲突。"""
        if len(self.participants) < 2:
            return {}

        all_keys = set()
        for knowledge in self.participants.values():
            all_keys.update(knowledge.keys())

        conflicts = {}
        for key in all_keys:
            values = []
            for node_id, knowledge in self.participants.items():
                if key in knowledge:
                    values.append((node_id, knowledge[key]))

            if len(values) >= 2:
                vals = [v for _, v in values]
                variance = sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / len(vals)
                if variance > 0.1:
                    conflicts[key] = {
                        'values': dict(values),
                        'variance': variance
                    }

        return conflicts


class ReputationSystem:
    """联邦学习信誉系统。"""

    def __init__(self):
        self.reputations = {}
        self.contributions = defaultdict(list)
        self.base_reputation = 50.0

    def register_node(self, node_id):
        """注册节点。"""
        self.reputations[node_id] = self.base_reputation

    def record_contribution(self, node_id, quality_score):
        """记录贡献质量。"""
        if node_id not in self.reputations:
            self.register_node(node_id)
        self.contributions[node_id].append(quality_score)
        # 更新信誉：指数移动平均
        alpha = 0.3
        old_rep = self.reputations[node_id]
        self.reputations[node_id] = alpha * quality_score + (1 - alpha) * old_rep

    def get_reputation(self, node_id):
        """获取节点信誉分。"""
        return self.reputations.get(node_id, 0.0)

    def get_top_nodes(self, k=5):
        """获取信誉最高的k个节点。"""
        sorted_nodes = sorted(self.reputations.items(),
                              key=lambda x: x[1], reverse=True)
        return sorted_nodes[:k]

    def penalize(self, node_id, penalty=10.0):
        """惩罚节点。"""
        if node_id in self.reputations:
            self.reputations[node_id] = max(0, self.reputations[node_id] - penalty)

    def is_trusted(self, node_id, threshold=30.0):
        """检查节点是否可信。"""
        return self.get_reputation(node_id) >= threshold


class ContributionProver:
    """贡献证明系统。"""

    def __init__(self):
        self.proofs = {}

    def generate_proof(self, node_id, model_update, data_samples, timestamp=None):
        """生成贡献证明。"""
        timestamp = timestamp or time.time()
        # 将模型更新和数据样本哈希
        update_str = str(model_update)
        data_str = str(data_samples)
        proof_input = f"{node_id}:{update_str}:{data_str}:{timestamp}"
        proof_hash = hashlib.sha256(proof_input.encode()).hexdigest()

        proof = {
            'node_id': node_id,
            'proof_hash': proof_hash,
            'timestamp': timestamp,
            'update_size': len(model_update) if isinstance(model_update, list) else 1,
            'num_samples': len(data_samples) if isinstance(data_samples, list) else 0,
        }
        self.proofs[node_id] = proof
        return proof

    def verify_proof(self, node_id, model_update, data_samples, timestamp=None):
        """验证贡献证明。"""
        if node_id not in self.proofs:
            return False

        stored_proof = self.proofs[node_id]
        timestamp = timestamp or stored_proof['timestamp']

        # 重新计算哈希
        update_str = str(model_update)
        data_str = str(data_samples)
        proof_input = f"{node_id}:{update_str}:{data_str}:{timestamp}"
        computed_hash = hashlib.sha256(proof_input.encode()).hexdigest()

        return computed_hash == stored_proof['proof_hash']

    def get_proof(self, node_id):
        """获取节点的贡献证明。"""
        return self.proofs.get(node_id)


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestEncryptedStore(unittest.TestCase):
    """测试加密存储和检索。"""

    def setUp(self):
        self.encryptor = SimpleEncryptor("test_secret_key")
        self.store = EncryptedMemoryStore(self.encryptor)

    def test_encrypted_store(self):
        """加密存储后应能正确检索。"""
        # 存储记忆
        self.store.store_memory("key1", "hello world", {"tag": "greeting"})
        self.store.store_memory("key2", "federated learning", {"tag": "ai"})

        # 检索记忆
        result1 = self.store.retrieve_memory("key1")
        self.assertEqual(result1, "hello world",
                         "应能正确检索加密存储的数据")

        result2 = self.store.retrieve_memory("key2")
        self.assertEqual(result2, "federated learning",
                         "应能正确检索第二条数据")

        # 检查元数据
        self.assertEqual(self.store.metadata["key1"]["tag"], "greeting")

        # 检查不存在的key
        result3 = self.store.retrieve_memory("nonexistent")
        self.assertIsNone(result3, "不存在的key应返回None")

    def test_encrypted_store_delete(self):
        """应能正确删除加密记忆。"""
        self.store.store_memory("key1", "to be deleted")
        self.assertTrue(self.store.has_memory("key1"))
        self.assertTrue(self.store.delete_memory("key1"))
        self.assertFalse(self.store.has_memory("key1"))
        self.assertIsNone(self.store.retrieve_memory("key1"))

    def test_encrypted_store_overwrite(self):
        """覆盖存储应更新数据。"""
        self.store.store_memory("key1", "original")
        self.store.store_memory("key1", "updated")
        result = self.store.retrieve_memory("key1")
        self.assertEqual(result, "updated",
                         "覆盖存储应更新数据")


class TestEncryptedVectorSearch(unittest.TestCase):
    """测试加密向量搜索精度。"""

    def setUp(self):
        self.index = VectorIndex(SimpleEncryptor("vector_search_key"))
        # 添加一些测试向量
        self.index.add_vector("v1", [1.0, 0.0, 0.0])
        self.index.add_vector("v2", [0.0, 1.0, 0.0])
        self.index.add_vector("v3", [0.9, 0.1, 0.0])  # 接近v1
        self.index.add_vector("v4", [0.0, 0.0, 1.0])
        self.index.add_vector("v5", [0.5, 0.5, 0.0])

    def test_encrypted_vector_search(self):
        """加密向量搜索应返回正确的结果。"""
        query = [1.0, 0.0, 0.0]
        results = self.index.search(query, top_k=3)

        # v1应排在最前面（完全匹配）
        self.assertEqual(results[0][0], "v1",
                         "完全匹配的向量应排在最前")
        # v3应排第二（接近v1）
        self.assertEqual(results[0][1], 1.0,
                         "完全匹配的相似度应为1.0")
        self.assertGreater(results[1][1], results[2][1],
                           "v3应比v5更接近查询向量")

    def test_encrypted_vector_roundtrip(self):
        """加密向量的存储和解密应保持数据完整性。"""
        original = [1.5, 2.3, -0.7, 4.2]
        self.index.add_vector("test_vec", original)
        decrypted = self.index.decrypt_vector("test_vec")
        for i in range(len(original)):
            self.assertAlmostEqual(original[i], decrypted[i], places=5,
                                   msg=f"向量第{i}维应保持一致")


class TestSyncScheduler(unittest.TestCase):
    """测试同步调度正确性。"""

    def setUp(self):
        self.scheduler = SyncScheduler(num_nodes=5)

    def test_sync_scheduler(self):
        """同步调度应正确协调节点。"""
        # 初始状态应一致
        self.assertTrue(self.scheduler.is_consistent())

        # 部分节点更新
        self.scheduler.local_update(0)
        self.scheduler.local_update(1)
        self.assertFalse(self.scheduler.is_consistent())

        # 同步
        synced = self.scheduler.sync_round()
        self.assertTrue(synced, "存在不一致时应执行同步")
        self.assertTrue(self.scheduler.is_consistent())
        self.assertEqual(self.scheduler.node_versions[0], 1)
        self.assertEqual(self.scheduler.node_versions[4], 1)

        # 再次同步（已一致，无需同步）
        synced = self.scheduler.sync_round()
        self.assertFalse(synced, "已一致时无需同步")

    def test_sync_log(self):
        """同步日志应正确记录操作。"""
        self.scheduler.local_update(0)
        self.scheduler.local_update(2)
        self.scheduler.sync_round()
        log = self.scheduler.get_sync_log()

        self.assertEqual(len(log), 3)  # 2 local + 1 sync
        self.assertEqual(log[0], ('local', 0, 1))
        self.assertEqual(log[1], ('local', 2, 1))
        self.assertEqual(log[2][0], 'sync')


class TestGlobalAdversarial(unittest.TestCase):
    """测试对抗样本存储和检索。"""

    def setUp(self):
        self.memory = AdversarialMemory()

    def test_global_adversarial(self):
        """应能正确存储和检索对抗样本。"""
        original = [1.0, 2.0, 3.0]
        adv_fgsm = [1.05, 2.05, 2.95]
        adv_pgd = [1.08, 1.92, 3.1]

        self.memory.store_adversarial(original, adv_fgsm, "cat", "FGSM")
        self.memory.store_adversarial(original, adv_pgd, "cat", "PGD")

        # 按攻击类型检索
        fgsm_samples = self.memory.retrieve_by_attack_type("FGSM")
        self.assertEqual(len(fgsm_samples), 1)
        self.assertEqual(fgsm_samples[0]['attack_type'], "FGSM")

        pgd_samples = self.memory.retrieve_by_attack_type("PGD")
        self.assertEqual(len(pgd_samples), 1)

        # 按扰动预算检索
        within_budget = self.memory.retrieve_within_budget(budget=0.2)
        self.assertEqual(len(within_budget), 2)

        # 验证扰动
        self.assertTrue(
            self.memory.validate_perturbation(original, adv_fgsm, budget=0.2)
        )


class TestKnowledgeDistillation(unittest.TestCase):
    """测试联邦知识蒸馏。"""

    def setUp(self):
        self.distiller = KnowledgeDistiller(num_participants=3)

    def test_knowledge_distillation(self):
        """知识蒸馏应正确聚合多节点知识。"""
        self.distiller.add_participant("node_a", {"accuracy": 0.85, "loss": 0.3})
        self.distiller.add_participant("node_b", {"accuracy": 0.90, "loss": 0.25})
        self.distiller.add_participant("node_c", {"accuracy": 0.80, "loss": 0.35})

        distilled = self.distiller.distill(aggregation='weighted_average')

        self.assertAlmostEqual(distilled['accuracy'], (0.85 + 0.90 + 0.80) / 3,
                               places=5, msg="蒸馏后的准确率应为平均值")
        self.assertAlmostEqual(distilled['loss'], (0.3 + 0.25 + 0.35) / 3,
                               places=5, msg="蒸馏后的损失应为平均值")

    def test_knowledge_conflict(self):
        """应能检测知识冲突。"""
        self.distiller.add_participant("node_a", {"param_w": 1.0, "param_b": 0.5})
        self.distiller.add_participant("node_b", {"param_w": 5.0, "param_b": 0.5})
        self.distiller.add_participant("node_c", {"param_w": 1.5, "param_b": 0.5})

        conflicts = self.distiller.get_knowledge_conflicts()

        # param_w有较大方差，应被检测为冲突
        self.assertIn('param_w', conflicts,
                      "param_w应被检测为冲突")
        self.assertGreater(conflicts['param_w']['variance'], 0.1,
                           "冲突参数的方差应大于阈值")

        # param_b值相同，不应有冲突
        self.assertNotIn('param_b', conflicts,
                         "param_b不应被检测为冲突")


class TestReputationSystem(unittest.TestCase):
    """测试信誉系统评分正确性。"""

    def setUp(self):
        self.reputation = ReputationSystem()

    def test_reputation_system(self):
        """信誉系统应正确计算和更新评分。"""
        self.reputation.register_node("node_a")
        self.reputation.register_node("node_b")

        # 记录贡献
        self.reputation.record_contribution("node_a", 80.0)
        self.reputation.record_contribution("node_a", 90.0)
        self.reputation.record_contribution("node_b", 40.0)

        rep_a = self.reputation.get_reputation("node_a")
        rep_b = self.reputation.get_reputation("node_b")

        # node_a有高质量贡献，信誉应更高
        self.assertGreater(rep_a, rep_b,
                           "高质量贡献的节点信誉应更高")
        self.assertGreater(rep_a, 50.0,
                           "多次高质量贡献应提升信誉")

        # 惩罚
        self.reputation.penalize("node_b", penalty=20.0)
        rep_b_after = self.reputation.get_reputation("node_b")
        self.assertLess(rep_b_after, rep_b,
                        "惩罚后信誉应降低")

        # 信任检查
        self.assertTrue(self.reputation.is_trusted("node_a"))
        self.assertFalse(self.reputation.is_trusted("node_b", threshold=30.0))

    def test_top_nodes(self):
        """应能正确返回信誉最高的节点。"""
        for i in range(10):
            self.reputation.register_node(f"node_{i}")
            self.reputation.record_contribution(f"node_{i}", random.uniform(0, 100))

        top_3 = self.reputation.get_top_nodes(k=3)
        self.assertEqual(len(top_3), 3)

        # 验证排序正确
        for i in range(len(top_3) - 1):
            self.assertGreaterEqual(top_3[i][1], top_3[i + 1][1],
                                    "信誉排名应降序排列")


class TestContributionProof(unittest.TestCase):
    """测试贡献证明验证。"""

    def setUp(self):
        self.prover = ContributionProver()

    def test_contribution_proof(self):
        """贡献证明应能正确生成和验证。"""
        node_id = "node_001"
        model_update = [0.1, 0.2, 0.3, 0.4]
        data_samples = [1, 2, 3, 4, 5]
        timestamp = 1234567890.0

        # 生成证明
        proof = self.prover.generate_proof(node_id, model_update, data_samples, timestamp)
        self.assertIsNotNone(proof)
        self.assertEqual(proof['node_id'], node_id)
        self.assertEqual(proof['num_samples'], 5)
        self.assertEqual(proof['update_size'], 4)

        # 验证证明
        is_valid = self.prover.verify_proof(node_id, model_update, data_samples, timestamp)
        self.assertTrue(is_valid, "正确的贡献应通过验证")

        # 篡改数据后验证应失败
        tampered_update = [0.1, 0.2, 0.3, 0.5]  # 最后一个值被篡改
        is_valid_tampered = self.prover.verify_proof(
            node_id, tampered_update, data_samples, timestamp
        )
        self.assertFalse(is_valid_tampered, "篡改的贡献应无法通过验证")

        # 不存在的节点
        is_valid_missing = self.prover.verify_proof(
            "nonexistent", model_update, data_samples, timestamp
        )
        self.assertFalse(is_valid_missing, "不存在的节点应无法通过验证")


if __name__ == '__main__':
    unittest.main()
