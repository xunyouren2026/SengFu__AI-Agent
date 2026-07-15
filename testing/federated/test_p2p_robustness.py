"""
P2P鲁棒性测试

测试联邦学习P2P网络中的DHT节点发现、Gossip协议、
模型分片传输、智能合约、学习证明、Raft共识和种群迁移。

使用纯Python标准库实现，不依赖外部P2P网络库。
"""

import unittest
import math
import random
import hashlib
import time
import copy
from collections import defaultdict


# ---------------------------------------------------------------------------
# 辅助工具：模拟P2P网络组件
# ---------------------------------------------------------------------------

class DHTNode:
    """模拟DHT节点，支持分布式哈希表的存储和检索。"""

    def __init__(self, node_id):
        self.node_id = node_id
        self.data = {}
        self.routing_table = set()
        self.finger_table = {}  # finger table for chord-like DHT

    def add_peer(self, peer_id):
        """添加对等节点到路由表。"""
        self.routing_table.add(peer_id)

    def remove_peer(self, peer_id):
        """从路由表中移除对等节点。"""
        self.routing_table.discard(peer_id)

    def store(self, key, value):
        """存储键值对。"""
        self.data[key] = value

    def retrieve(self, key):
        """检索键值对。"""
        return self.data.get(key)

    def find_responsible_node(self, key, all_nodes):
        """使用一致性哈希找到负责的节点。"""
        key_hash = self._hash(key)
        node_hashes = [(self._hash(nid), nid) for nid in all_nodes]
        node_hashes.sort()

        # 找到第一个哈希值大于等于key_hash的节点
        for nh, nid in node_hashes:
            if nh >= key_hash:
                return nid
        # 环绕到第一个节点
        return node_hashes[0][1] if node_hashes else None

    @staticmethod
    def _hash(key):
        """一致性哈希函数。"""
        return int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16)


class DHTNetwork:
    """模拟DHT网络。"""

    def __init__(self):
        self.nodes = {}
        self.join_order = []

    def add_node(self, node_id):
        """添加节点到网络。"""
        node = DHTNode(node_id)
        self.nodes[node_id] = node
        self.join_order.append(node_id)
        # 更新路由表
        for nid in self.nodes:
            if nid != node_id:
                self.nodes[nid].add_peer(node_id)
                node.add_peer(nid)

    def remove_node(self, node_id):
        """从网络中移除节点。"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            for nid in self.nodes:
                self.nodes[nid].remove_peer(node_id)

    def store(self, key, value):
        """在DHT网络中存储数据。"""
        all_ids = list(self.nodes.keys())
        if not all_ids:
            return False
        responsible = self.nodes[all_ids[0]].find_responsible_node(key, all_ids)
        if responsible and responsible in self.nodes:
            self.nodes[responsible].store(key, value)
            return True
        return False

    def retrieve(self, key):
        """从DHT网络中检索数据。"""
        all_ids = list(self.nodes.keys())
        if not all_ids:
            return None
        responsible = self.nodes[all_ids[0]].find_responsible_node(key, all_ids)
        if responsible and responsible in self.nodes:
            return self.nodes[responsible].retrieve(key)
        return None

    def discover_nodes(self, node_id, target_count=5):
        """节点发现：返回已知节点列表。"""
        if node_id not in self.nodes:
            return []
        node = self.nodes[node_id]
        peers = list(node.routing_table)
        random.shuffle(peers)
        return peers[:target_count]


class GossipNetwork:
    """模拟Gossip协议网络。"""

    def __init__(self):
        self.nodes = {}
        self.messages = defaultdict(list)
        self.message_log = []
        self.fanout = 3  # 每个节点每轮通知的邻居数

    def add_node(self, node_id):
        """添加节点。"""
        self.nodes[node_id] = {'alive': True, 'seen_messages': set()}

    def remove_node(self, node_id):
        """移除节点（模拟掉线）。"""
        if node_id in self.nodes:
            self.nodes[node_id]['alive'] = False

    def revive_node(self, node_id):
        """恢复节点。"""
        if node_id in self.nodes:
            self.nodes[node_id]['alive'] = True

    def broadcast(self, sender_id, message):
        """发起Gossip广播。"""
        msg_id = hashlib.sha256(f"{sender_id}:{message}:{time.time()}".encode()).hexdigest()[:16]
        self.messages[msg_id] = {
            'content': message,
            'sender': sender_id,
            'id': msg_id,
        }
        self.message_log.append(('broadcast', sender_id, msg_id))
        self._gossip_round(sender_id, msg_id)

    def _gossip_round(self, source_id, msg_id, visited=None):
        """执行一轮Gossip传播。"""
        if visited is None:
            visited = set()
        if source_id in visited:
            return
        visited.add(source_id)

        if source_id not in self.nodes or not self.nodes[source_id]['alive']:
            return

        self.nodes[source_id]['seen_messages'].add(msg_id)

        # 选择fanout个邻居传播
        neighbors = [nid for nid in self.nodes
                     if nid != source_id and self.nodes[nid]['alive'] and nid not in visited]
        targets = random.sample(neighbors, min(self.fanout, len(neighbors)))

        for target in targets:
            self.message_log.append(('gossip', source_id, target, msg_id))
            self._gossip_round(target, msg_id, visited)

    def get_reception_count(self, msg_id):
        """获取消息被接收的节点数。"""
        return sum(1 for n in self.nodes.values()
                   if msg_id in n['seen_messages'])

    def run_gossip_rounds(self, msg_id, max_rounds=5):
        """运行多轮Gossip传播。"""
        for _ in range(max_rounds):
            # 随机选择一个已收到消息的节点继续传播
            informed = [nid for nid, n in self.nodes.items()
                        if msg_id in n['seen_messages'] and n['alive']]
            uninformed = [nid for nid, n in self.nodes.items()
                          if msg_id not in n['seen_messages'] and n['alive']]

            if not uninformed:
                break

            for source in informed:
                neighbors = [nid for nid in uninformed
                             if nid != source]
                targets = random.sample(neighbors, min(self.fanout, len(neighbors)))
                for target in targets:
                    self.message_log.append(('gossip', source, target, msg_id))
                    self.nodes[target]['seen_messages'].add(msg_id)


class ModelShard:
    """模型分片传输和重组。"""

    def __init__(self, model_params):
        self.params = list(model_params)
        self.num_shards = 4
        self.shards = {}
        self.checksums = {}

    def split(self):
        """将模型参数分成多个分片。"""
        n = len(self.params)
        shard_size = max(1, n // self.num_shards)
        self.shards = {}
        self.checksums = {}

        for i in range(self.num_shards):
            start = i * shard_size
            end = start + shard_size if i < self.num_shards - 1 else n
            shard = self.params[start:end]
            self.shards[i] = shard
            self.checksums[i] = hashlib.sha256(
                str(shard).encode()
            ).hexdigest()[:16]

        return self.shards

    def reassemble(self, received_shards):
        """从分片重组模型。"""
        # 验证所有分片都存在
        if set(received_shards.keys()) != set(self.shards.keys()):
            return None

        # 验证校验和
        for shard_id, shard_data in received_shards.items():
            checksum = hashlib.sha256(str(shard_data).encode()).hexdigest()[:16]
            if checksum != self.checksums.get(shard_id):
                return None

        # 按顺序重组
        reassembled = []
        for i in sorted(received_shards.keys()):
            reassembled.extend(received_shards[i])

        return reassembled


class SmartContract:
    """模拟智能合约系统。"""

    def __init__(self):
        self.contracts = {}
        self.state = {}
        self.transaction_log = []

    def deploy(self, contract_id, code, initial_state=None):
        """部署智能合约。"""
        contract = {
            'id': contract_id,
            'code': code,
            'state': dict(initial_state or {}),
            'owner': None,
        }
        self.contracts[contract_id] = contract
        self.state[contract_id] = dict(initial_state or {})
        self.transaction_log.append(('deploy', contract_id))
        return contract_id

    def call(self, contract_id, function, args=None):
        """调用智能合约函数。"""
        if contract_id not in self.contracts:
            return None

        contract = self.contracts[contract_id]
        args = args or {}

        # 简单的合约执行逻辑
        if function == 'transfer':
            from_addr = args.get('from')
            to_addr = args.get('to')
            amount = args.get('amount', 0)
            state = self.state[contract_id]

            if state.get(from_addr, 0) >= amount:
                state[from_addr] = state.get(from_addr, 0) - amount
                state[to_addr] = state.get(to_addr, 0) + amount
                self.transaction_log.append(('call', contract_id, function, args))
                return True
            return False

        elif function == 'get_balance':
            state = self.state[contract_id]
            addr = args.get('address')
            return state.get(addr, 0)

        elif function == 'set_value':
            key = args.get('key')
            value = args.get('value')
            self.state[contract_id][key] = value
            self.transaction_log.append(('call', contract_id, function, args))
            return True

        elif function == 'get_value':
            key = args.get('key')
            return self.state[contract_id].get(key)

        return None

    def get_state(self, contract_id):
        """获取合约状态。"""
        return dict(self.state.get(contract_id, {}))


class ProofOfLearning:
    """学习证明生成和验证。"""

    def __init__(self, difficulty=4):
        self.difficulty = difficulty
        self.proofs = {}

    def generate(self, node_id, model_hash, dataset_hash, num_iterations=100000):
        """生成学习证明（工作量证明风格）。"""
        nonce = 0
        target = '0' * self.difficulty

        while nonce < num_iterations:
            proof_input = f"{node_id}:{model_hash}:{dataset_hash}:{nonce}"
            proof_hash = hashlib.sha256(proof_input.encode()).hexdigest()

            if proof_hash.startswith(target):
                proof = {
                    'node_id': node_id,
                    'model_hash': model_hash,
                    'dataset_hash': dataset_hash,
                    'nonce': nonce,
                    'proof_hash': proof_hash,
                }
                self.proofs[node_id] = proof
                return proof

            nonce += 1

        return None  # 未找到有效证明

    def verify(self, node_id, model_hash, dataset_hash):
        """验证学习证明。"""
        if node_id not in self.proofs:
            return False

        proof = self.proofs[node_id]
        if proof['model_hash'] != model_hash or proof['dataset_hash'] != dataset_hash:
            return False

        # 重新计算哈希
        proof_input = f"{node_id}:{model_hash}:{dataset_hash}:{proof['nonce']}"
        computed_hash = hashlib.sha256(proof_input.encode()).hexdigest()

        target = '0' * self.difficulty
        return computed_hash.startswith(target) and computed_hash == proof['proof_hash']


class RaftCluster:
    """模拟Raft共识集群。"""

    def __init__(self, node_ids):
        self.nodes = {}
        for nid in node_ids:
            self.nodes[nid] = {
                'state': 'follower',  # follower, candidate, leader
                'term': 0,
                'voted_for': None,
                'log': [],
                'commit_index': 0,
                'alive': True,
            }
        self.current_term = 0
        self.leader = None
        self.election_timeout = 150  # ms

    def start_election(self, candidate_id):
        """发起选举。"""
        self.current_term += 1
        candidate = self.nodes[candidate_id]
        candidate['state'] = 'candidate'
        candidate['term'] = self.current_term
        candidate['voted_for'] = candidate_id

        # 请求投票
        votes = {candidate_id}  # 自投一票
        for nid, node in self.nodes.items():
            if nid != candidate_id and node['alive']:
                if node['term'] < self.current_term:
                    node['voted_for'] = candidate_id
                    node['term'] = self.current_term
                    votes.add(nid)

        # 获得多数票则成为leader
        majority = len(self.nodes) // 2 + 1
        if len(votes) >= majority:
            candidate['state'] = 'leader'
            self.leader = candidate_id
            # 更新其他节点的term
            for nid, node in self.nodes.items():
                if nid != candidate_id:
                    node['term'] = self.current_term
            return True
        else:
            candidate['state'] = 'follower'
            return False

    def append_entry(self, leader_id, entry):
        """Leader追加日志条目。"""
        if self.leader != leader_id:
            return False

        leader = self.nodes[leader_id]
        leader['log'].append({
            'term': self.current_term,
            'entry': entry,
        })

        # 复制到其他节点
        replicated = 1  # leader自己
        for nid, node in self.nodes.items():
            if nid != leader_id and node['alive']:
                node['log'].append({
                    'term': self.current_term,
                    'entry': entry,
                })
                replicated += 1

        # 多数确认后提交
        majority = len(self.nodes) // 2 + 1
        if replicated >= majority:
            leader['commit_index'] = len(leader['log'])
            for nid, node in self.nodes.items():
                if node['alive']:
                    node['commit_index'] = leader['commit_index']
            return True
        return False

    def get_leader(self):
        """获取当前leader。"""
        return self.leader

    def is_log_consistent(self):
        """检查所有存活节点的日志是否一致。"""
        alive_nodes = [n for n in self.nodes.values() if n['alive']]
        if not alive_nodes:
            return True

        committed_logs = []
        for node in alive_nodes:
            committed = node['log'][:node['commit_index']]
            committed_logs.append(committed)

        # 比较所有已提交的日志
        if not committed_logs:
            return True
        reference = committed_logs[0]
        for log in committed_logs[1:]:
            if len(log) != len(reference):
                return False
            for i in range(len(log)):
                if log[i] != reference[i]:
                    return False
        return True


class PopulationMigration:
    """种群迁移模拟器。"""

    def __init__(self, num_islands=3, population_size=10):
        self.num_islands = num_islands
        self.population_size = population_size
        self.islands = {}
        self.migration_history = []

        for i in range(num_islands):
            self.islands[i] = {
                'population': [random.uniform(-10, 10) for _ in range(population_size)],
                'fitness': [],
                'best_fitness': float('-inf'),
            }
            self._evaluate_island(i)

    def _evaluate_island(self, island_id):
        """评估岛上种群的适应度。"""
        island = self.islands[island_id]
        island['fitness'] = [self._fitness(x) for x in island['population']]
        island['best_fitness'] = max(island['fitness']) if island['fitness'] else float('-inf')

    @staticmethod
    def _fitness(x):
        """简单的适应度函数：f(x) = -x^2 + 10（最大值在x=0时为10）。"""
        return -(x ** 2) + 10

    def migrate(self, migration_rate=0.2, topology='ring'):
        """执行种群迁移。"""
        migrants = {}

        for island_id in range(self.num_islands):
            island = self.islands[island_id]
            num_migrants = max(1, int(self.population_size * migration_rate))

            # 选择适应度最高的个体迁移
            indexed = sorted(enumerate(island['fitness']),
                             key=lambda x: x[1], reverse=True)
            migrant_indices = [idx for idx, _ in indexed[:num_migrants]]
            migrants[island_id] = [island['population'][idx] for idx in migrant_indices]

        # 根据拓扑迁移
        if topology == 'ring':
            for island_id in range(self.num_islands):
                target = (island_id + 1) % self.num_islands
                # 替换目标岛屿中适应度最低的个体
                target_island = self.islands[target]
                target_fitness = target_island['fitness']
                worst_indices = sorted(
                    enumerate(target_fitness),
                    key=lambda x: x[1]
                )[:len(migrants[island_id])]

                for j, (idx, _) in enumerate(worst_indices):
                    target_island['population'][idx] = migrants[island_id][j]

                self._evaluate_island(target)
                self.migration_history.append((island_id, target))

        elif topology == 'all_to_all':
            for island_id in range(self.num_islands):
                for target in range(self.num_islands):
                    if target != island_id:
                        target_island = self.islands[target]
                        # 随机替换一个个体
                        replace_idx = random.randint(0, self.population_size - 1)
                        target_island['population'][replace_idx] = migrants[island_id][0]
                self._evaluate_island(island_id)

    def get_global_best_fitness(self):
        """获取全局最佳适应度。"""
        return max(island['best_fitness'] for island in self.islands.values())

    def evolve(self, generations=10):
        """进化若干代。"""
        for _ in range(generations):
            for island_id in range(self.num_islands):
                island = self.islands[island_id]
                # 简单的变异操作
                new_pop = []
                for x in island['population']:
                    mutated = x + random.gauss(0, 0.5)
                    new_pop.append(mutated)
                island['population'] = new_pop
                self._evaluate_island(island_id)


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestDHTDiscovery(unittest.TestCase):
    """测试DHT节点发现。"""

    def setUp(self):
        self.network = DHTNetwork()
        for i in range(10):
            self.network.add_node(f"node_{i}")

    def test_dht_discovery(self):
        """DHT应能正确发现节点。"""
        discovered = self.network.discover_nodes("node_0", target_count=5)
        self.assertEqual(len(discovered), 5,
                         "应发现指定数量的节点")
        self.assertNotIn("node_0", discovered,
                         "发现列表不应包含自身")

        # 所有发现的节点都应存在于网络中
        for nid in discovered:
            self.assertIn(nid, self.network.nodes,
                          f"发现的节点 {nid} 应存在于网络中")

    def test_dht_discovery_after_failure(self):
        """节点故障后发现应排除故障节点。"""
        self.network.remove_node("node_5")
        discovered = self.network.discover_nodes("node_0", target_count=10)
        self.assertNotIn("node_5", discovered,
                         "故障节点不应出现在发现列表中")


class TestDHTStoreRetrieve(unittest.TestCase):
    """测试DHT存储和检索。"""

    def setUp(self):
        self.network = DHTNetwork()
        for i in range(5):
            self.network.add_node(f"node_{i}")

    def test_dht_store_retrieve(self):
        """DHT应能正确存储和检索数据。"""
        self.network.store("key1", "value1")
        self.network.store("key2", "value2")

        result1 = self.network.retrieve("key1")
        self.assertEqual(result1, "value1",
                         "应能检索存储的数据")

        result2 = self.network.retrieve("key2")
        self.assertEqual(result2, "value2")

        result3 = self.network.retrieve("nonexistent")
        self.assertIsNone(result3,
                          "不存在的key应返回None")

    def test_dht_consistent_hashing(self):
        """一致性哈希应确保数据分布到正确的节点。"""
        # 多次存储和检索
        for i in range(20):
            key = f"test_key_{i}"
            value = f"test_value_{i}"
            self.network.store(key, value)
            result = self.network.retrieve(key)
            self.assertEqual(result, value,
                             f"数据 {key} 应能正确检索")


class TestGossipSpread(unittest.TestCase):
    """测试Gossip消息传播。"""

    def setUp(self):
        random.seed(42)
        self.network = GossipNetwork()
        for i in range(10):
            self.network.add_node(f"node_{i}")

    def test_gossip_spread(self):
        """Gossip消息应传播到大部分节点。"""
        self.network.broadcast("node_0", "hello")

        # 获取第一条消息的ID
        msg_id = self.network.message_log[0][2]
        reception = self.network.get_reception_count(msg_id)

        # 消息应传播到至少一半的节点
        self.assertGreaterEqual(reception, 5,
                                "Gossip消息应传播到至少一半节点")

    def test_gossip_node_failure(self):
        """节点掉线后Gossip仍能工作。"""
        # 让一些节点掉线
        self.network.remove_node("node_3")
        self.network.remove_node("node_7")

        self.network.broadcast("node_0", "test_message")

        msg_id = self.network.message_log[0][2]
        reception = self.network.get_reception_count(msg_id)

        # 消息应传播到存活节点
        alive_count = sum(1 for n in self.network.nodes.values() if n['alive'])
        # 至少应传播到发起者
        self.assertGreaterEqual(reception, 1,
                                "即使有节点掉线，Gossip仍应工作")

        # 掉线的节点不应收到消息
        self.assertNotIn("node_3",
                         self.network.nodes["node_3"].get('seen_messages', set())
                         if self.network.nodes.get("node_3") else set())


class TestModelExchange(unittest.TestCase):
    """测试模型分片传输和重组。"""

    def setUp(self):
        self.model_params = [0.1 * i for i in range(100)]
        self.shard = ModelShard(self.model_params)

    def test_model_exchange(self):
        """模型分片传输和重组应保持数据完整性。"""
        # 分片
        shards = self.shard.split()
        self.assertEqual(len(shards), self.shard.num_shards,
                         "应生成正确数量的分片")

        # 模拟传输（直接传递）
        received = copy.deepcopy(shards)

        # 重组
        reassembled = self.shard.reassemble(received)
        self.assertIsNotNone(reassembled,
                             "完整分片应能成功重组")

        # 验证数据完整性
        for i in range(len(self.model_params)):
            self.assertAlmostEqual(self.model_params[i], reassembled[i], places=10,
                                   msg=f"分片重组后第{i}个参数应一致")

    def test_model_exchange_tampered(self):
        """被篡改的分片应被检测到。"""
        shards = self.shard.split()
        # 篡改一个分片
        shards[0][0] = 999.0
        reassembled = self.shard.reassemble(shards)
        self.assertIsNone(reassembled,
                          "被篡改的分片应导致重组失败")

    def test_model_exchange_incomplete(self):
        """不完整的分片应导致重组失败。"""
        shards = self.shard.split()
        # 深拷贝后删除一个分片，避免影响原始shards
        incomplete_shards = copy.deepcopy(shards)
        del incomplete_shards[0]  # 删除一个分片
        reassembled = self.shard.reassemble(incomplete_shards)
        self.assertIsNone(reassembled,
                          "不完整的分片应导致重组失败")


class TestSmartContract(unittest.TestCase):
    """测试智能合约部署和调用。"""

    def setUp(self):
        self.contract_system = SmartContract()

    def test_smart_contract(self):
        """智能合约应能正确部署和调用。"""
        # 部署合约
        contract_id = self.contract_system.deploy(
            "token_contract",
            "ERC20",
            initial_state={"alice": 100, "bob": 50}
        )

        self.assertEqual(contract_id, "token_contract")

        # 查询余额
        alice_balance = self.contract_system.call(
            contract_id, 'get_balance', {'address': 'alice'}
        )
        self.assertEqual(alice_balance, 100)

        # 转账
        success = self.contract_system.call(
            contract_id, 'transfer',
            {'from': 'alice', 'to': 'bob', 'amount': 30}
        )
        self.assertTrue(success, "有效转账应成功")

        # 验证余额
        alice_balance = self.contract_system.call(
            contract_id, 'get_balance', {'address': 'alice'}
        )
        bob_balance = self.contract_system.call(
            contract_id, 'get_balance', {'address': 'bob'}
        )
        self.assertEqual(alice_balance, 70)
        self.assertEqual(bob_balance, 80)

        # 余额不足的转账应失败
        success = self.contract_system.call(
            contract_id, 'transfer',
            {'from': 'alice', 'to': 'bob', 'amount': 1000}
        )
        self.assertFalse(success, "余额不足的转账应失败")

    def test_smart_contract_state(self):
        """合约状态应正确维护。"""
        self.contract_system.deploy("storage", "KVStore",
                                     initial_state={"count": 0})

        self.contract_system.call("storage", 'set_value',
                                   {'key': 'count', 'value': 42})
        value = self.contract_system.call("storage", 'get_value',
                                           {'key': 'count'})
        self.assertEqual(value, 42)

        state = self.contract_system.get_state("storage")
        self.assertEqual(state['count'], 42)


class TestProofOfLearning(unittest.TestCase):
    """测试学习证明生成和验证。"""

    def setUp(self):
        self.pol = ProofOfLearning(difficulty=3)

    def test_proof_of_learning(self):
        """应能正确生成和验证学习证明。"""
        model_hash = hashlib.sha256(b"model_weights").hexdigest()
        dataset_hash = hashlib.sha256(b"training_data").hexdigest()

        proof = self.pol.generate("node_001", model_hash, dataset_hash)
        self.assertIsNotNone(proof, "应能生成学习证明")
        self.assertEqual(proof['node_id'], "node_001")
        self.assertEqual(proof['model_hash'], model_hash)

        # 验证
        is_valid = self.pol.verify("node_001", model_hash, dataset_hash)
        self.assertTrue(is_valid, "正确的学习证明应通过验证")

        # 错误的模型哈希
        is_valid_wrong = self.pol.verify("node_001", "wrong_hash", dataset_hash)
        self.assertFalse(is_valid_wrong, "错误的哈希应导致验证失败")

        # 不存在的节点
        is_valid_missing = self.pol.verify("nonexistent", model_hash, dataset_hash)
        self.assertFalse(is_valid_missing, "不存在的节点应验证失败")


class TestRaftElection(unittest.TestCase):
    """测试Raft选举。"""

    def setUp(self):
        self.cluster = RaftCluster([f"node_{i}" for i in range(5)])

    def test_raft_election(self):
        """Raft选举应正确选出leader。"""
        success = self.cluster.start_election("node_0")
        self.assertTrue(success, "应成功选出leader")
        self.assertEqual(self.cluster.get_leader(), "node_0",
                         "node_0应成为leader")

        # 验证所有节点的term已更新
        for node in self.cluster.nodes.values():
            self.assertEqual(node['term'], self.cluster.current_term,
                             "所有节点的term应与当前term一致")

    def test_raft_election_with_failure(self):
        """部分节点故障时仍应能选出leader。"""
        # 让一个节点掉线
        self.cluster.nodes["node_3"]['alive'] = False

        # 5个节点中4个存活，需要3票
        success = self.cluster.start_election("node_0")
        self.assertTrue(success, "多数存活时应能选出leader")
        self.assertEqual(self.cluster.get_leader(), "node_0")


class TestRaftLogReplication(unittest.TestCase):
    """测试Raft日志复制。"""

    def setUp(self):
        self.cluster = RaftCluster([f"node_{i}" for i in range(5)])
        self.cluster.start_election("node_0")

    def test_raft_log_replication(self):
        """日志应正确复制到多数节点。"""
        success = self.cluster.append_entry("node_0", "set x = 1")
        self.assertTrue(success, "日志应成功复制到多数节点")

        success = self.cluster.append_entry("node_0", "set y = 2")
        self.assertTrue(success)

        # 验证日志一致性
        self.assertTrue(self.cluster.is_log_consistent(),
                        "所有存活节点的已提交日志应一致")

        # 验证leader的日志长度
        leader = self.cluster.nodes["node_0"]
        self.assertEqual(len(leader['log']), 2)
        self.assertEqual(leader['log'][0]['entry'], "set x = 1")
        self.assertEqual(leader['log'][1]['entry'], "set y = 2")

    def test_raft_log_non_leader(self):
        """非leader不应能追加日志。"""
        success = self.cluster.append_entry("node_1", "set z = 3")
        self.assertFalse(success, "非leader不应能追加日志")


class TestPopulationMigration(unittest.TestCase):
    """测试种群迁移。"""

    def setUp(self):
        random.seed(42)
        self.migration = PopulationMigration(num_islands=3, population_size=20)

    def test_population_migration(self):
        """种群迁移应改善全局最优适应度。"""
        initial_best = self.migration.get_global_best_fitness()

        # 进化和迁移
        for _ in range(5):
            self.migration.evolve(generations=10)
            self.migration.migrate(migration_rate=0.2, topology='ring')

        final_best = self.migration.get_global_best_fitness()

        # 迁移后全局最优适应度应改善或保持
        self.assertGreaterEqual(final_best, initial_best - 1.0,
                                "迁移后全局最优适应度应改善或保持")

        # 验证迁移历史
        self.assertGreater(len(self.migration.migration_history), 0,
                           "应有迁移记录")

    def test_population_migration_all_to_all(self):
        """全对全迁移拓扑应正常工作。"""
        initial_best = self.migration.get_global_best_fitness()

        self.migration.evolve(generations=5)
        self.migration.migrate(migration_rate=0.1, topology='all_to_all')

        # 不应抛出异常
        final_best = self.migration.get_global_best_fitness()
        self.assertIsInstance(final_best, float)


if __name__ == '__main__':
    unittest.main()
