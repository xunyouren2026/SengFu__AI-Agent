"""
联邦知识蒸馏 - Agent间交换软标签而非数据

实现联邦知识蒸馏(Federated Knowledge Distillation)，允许多个Agent
在不共享原始数据的情况下，通过交换软标签(概率分布)进行协作学习。
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import random
import math
import hashlib


@dataclass
class SoftLabel:
    """软标签 - 知识蒸馏中的概率分布"""
    logits: Dict[int, float]  # 类别 -> logit值
    temperature: float = 1.0
    
    def get_probabilities(self) -> Dict[int, float]:
        """获取概率分布"""
        exp_logits = {k: math.exp(v / self.temperature) 
                     for k, v in self.logits.items()}
        total = sum(exp_logits.values())
        return {k: v / total for k, v in exp_logits.items()}
    
    def get_entropy(self) -> float:
        """计算分布的熵"""
        probs = self.get_probabilities()
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log(p)
        return entropy


@dataclass
class KnowledgePacket:
    """知识包 - Agent间交换的知识单元"""
    packet_id: str
    agent_id: str
    input_hash: str  # 输入数据的哈希(不传输实际数据)
    soft_label: SoftLabel
    
    # 元数据
    timestamp: float = field(default_factory=lambda: random.random() * 1000)
    confidence: float = 1.0
    version: int = 1
    
    # 标签
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalModel:
    """本地模型表示"""
    agent_id: str
    model_version: int = 1
    parameters: Dict[str, float] = field(default_factory=dict)
    
    def predict_logits(self, input_data: Any) -> Dict[int, float]:
        """预测logits"""
        # 简化的线性模型
        if isinstance(input_data, (list, tuple)):
            logits = {}
            for i in range(10):  # 假设10个类别
                logit = sum(self.parameters.get(f'w{i}_{j}', 0.1) * x 
                           for j, x in enumerate(input_data))
                logits[i] = logit
            return logits
        return {0: 0.0}


class FederatedKnowledgeDistiller:
    """联邦知识蒸馏器"""
    
    def __init__(self, agent_id: str, temperature: float = 2.0):
        self.agent_id = agent_id
        self.temperature = temperature
        self.local_model = LocalModel(agent_id=agent_id)
        
        # 知识存储
        self.local_knowledge: Dict[str, KnowledgePacket] = {}
        self.received_knowledge: Dict[str, List[KnowledgePacket]] = defaultdict(list)
        
        # 邻居Agent
        self.neighbors: Set[str] = set()
        self.neighbor_weights: Dict[str, float] = {}
        
        # 学习参数
        self.learning_rate = 0.01
        self.distillation_weight = 0.7
        
    def compute_soft_label(self, input_data: Any, 
                          return_logits: bool = False) -> SoftLabel:
        """计算输入数据的软标签"""
        logits = self.local_model.predict_logits(input_data)
        soft_label = SoftLabel(logits=logits, temperature=self.temperature)
        
        if return_logits:
            return soft_label, logits
        return soft_label
    
    def create_knowledge_packet(self, input_data: Any, 
                                input_id: Optional[str] = None) -> KnowledgePacket:
        """创建知识包"""
        # 计算输入哈希
        input_hash = self._hash_input(input_data, input_id)
        
        # 计算软标签
        soft_label = self.compute_soft_label(input_data)
        
        # 创建包
        packet = KnowledgePacket(
            packet_id=f"{self.agent_id}_{input_hash}_{random.randint(0, 10000)}",
            agent_id=self.agent_id,
            input_hash=input_hash,
            soft_label=soft_label,
            confidence=self._compute_confidence(soft_label),
            version=self.local_model.model_version
        )
        
        # 存储本地知识
        self.local_knowledge[input_hash] = packet
        
        return packet
    
    def _hash_input(self, input_data: Any, input_id: Optional[str] = None) -> str:
        """哈希输入数据"""
        if input_id:
            return input_id
        
        # 简化的哈希
        if isinstance(input_data, (list, tuple)):
            data_str = ','.join(str(x) for x in input_data)
        elif isinstance(input_data, dict):
            data_str = ','.join(f"{k}:{v}" for k, v in sorted(input_data.items()))
        else:
            data_str = str(input_data)
        
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def _compute_confidence(self, soft_label: SoftLabel) -> float:
        """计算软标签的置信度"""
        probs = soft_label.get_probabilities()
        max_prob = max(probs.values())
        
        # 熵越低，置信度越高
        entropy = soft_label.get_entropy()
        max_entropy = math.log(len(probs))
        
        confidence = 0.6 * max_prob + 0.4 * (1 - entropy / max_entropy)
        return confidence
    
    def receive_knowledge(self, packet: KnowledgePacket):
        """接收来自其他Agent的知识包"""
        if packet.agent_id == self.agent_id:
            return
        
        # 存储接收的知识
        self.received_knowledge[packet.input_hash].append(packet)
        
        # 更新邻居权重
        if packet.agent_id not in self.neighbors:
            self.neighbors.add(packet.agent_id)
            self.neighbor_weights[packet.agent_id] = 1.0
    
    def aggregate_knowledge(self, input_hash: str) -> Optional[SoftLabel]:
        """聚合多个Agent的软标签"""
        packets = self.received_knowledge.get(input_hash, [])
        
        if not packets:
            return None
        
        # 获取本地知识
        local_packet = self.local_knowledge.get(input_hash)
        if local_packet:
            packets = packets + [local_packet]
        
        # 加权聚合
        aggregated_logits: Dict[int, float] = defaultdict(float)
        total_weight = 0.0
        
        for packet in packets:
            weight = self.neighbor_weights.get(packet.agent_id, 1.0)
            weight *= packet.confidence
            
            for class_id, logit in packet.soft_label.logits.items():
                aggregated_logits[class_id] += weight * logit
            
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            aggregated_logits = {k: v / total_weight 
                               for k, v in aggregated_logits.items()}
        
        return SoftLabel(logits=aggregated_logits, temperature=self.temperature)
    
    def distill(self, input_data: Any, true_label: Optional[int] = None) -> Dict[str, Any]:
        """
        执行知识蒸馏
        
        结合本地预测和来自其他Agent的软标签进行学习
        """
        input_hash = self._hash_input(input_data)
        
        # 本地预测
        local_soft_label, local_logits = self.compute_soft_label(input_data, return_logits=True)
        
        # 获取聚合知识
        aggregated_soft_label = self.aggregate_knowledge(input_hash)
        
        # 计算蒸馏损失
        if aggregated_soft_label:
            # KL散度损失
            distillation_loss = self._kl_divergence(
                aggregated_soft_label.get_probabilities(),
                local_soft_label.get_probabilities()
            )
        else:
            distillation_loss = 0.0
        
        # 硬标签损失
        hard_loss = 0.0
        if true_label is not None:
            probs = local_soft_label.get_probabilities()
            hard_loss = -math.log(probs.get(true_label, 1e-10) + 1e-10)
        
        # 组合损失
        total_loss = (self.distillation_weight * distillation_loss + 
                     (1 - self.distillation_weight) * hard_loss)
        
        # 更新模型
        self._update_model(input_data, aggregated_soft_label, true_label, total_loss)
        
        return {
            'loss': total_loss,
            'distillation_loss': distillation_loss,
            'hard_loss': hard_loss,
            'local_confidence': self._compute_confidence(local_soft_label),
            'aggregated_knowledge_available': aggregated_soft_label is not None
        }
    
    def _kl_divergence(self, p: Dict[int, float], q: Dict[int, float]) -> float:
        """计算KL散度"""
        kl = 0.0
        for k, prob_p in p.items():
            prob_q = q.get(k, 1e-10)
            if prob_p > 0:
                kl += prob_p * math.log(prob_p / max(prob_q, 1e-10))
        return kl
    
    def _update_model(self, input_data: Any, 
                     target_soft_label: Optional[SoftLabel],
                     true_label: Optional[int],
                     loss: float):
        """更新本地模型"""
        if not isinstance(input_data, (list, tuple)):
            return
        
        # 简化的梯度下降更新
        for i, x in enumerate(input_data):
            for class_id in range(10):
                key = f'w{class_id}_{i}'
                current = self.local_model.parameters.get(key, 0.1)
                
                # 计算梯度
                gradient = loss * x * 0.01
                
                self.local_model.parameters[key] = current - self.learning_rate * gradient
    
    def update_neighbor_weight(self, neighbor_id: str, performance_delta: float):
        """更新邻居权重"""
        current = self.neighbor_weights.get(neighbor_id, 1.0)
        new_weight = current + 0.1 * performance_delta
        self.neighbor_weights[neighbor_id] = max(0.1, min(2.0, new_weight))


class FederatedKnowledgeServer:
    """联邦知识服务器 - 协调多个Agent的知识交换"""
    
    def __init__(self):
        self.registered_agents: Set[str] = set()
        self.knowledge_store: Dict[str, List[KnowledgePacket]] = defaultdict(list)
        self.agent_statistics: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self, agent_id: str):
        """注册Agent"""
        self.registered_agents.add(agent_id)
        self.agent_statistics[agent_id] = {
            'packets_sent': 0,
            'packets_received': 0,
            'total_knowledge_shared': 0
        }
    
    def submit_knowledge(self, packet: KnowledgePacket) -> bool:
        """提交知识包"""
        if packet.agent_id not in self.registered_agents:
            return False
        
        # 存储知识
        self.knowledge_store[packet.input_hash].append(packet)
        
        # 更新统计
        self.agent_statistics[packet.agent_id]['packets_sent'] += 1
        
        return True
    
    def request_knowledge(self, agent_id: str, input_hash: str,
                         exclude_self: bool = True) -> List[KnowledgePacket]:
        """请求知识"""
        if agent_id not in self.registered_agents:
            return []
        
        packets = self.knowledge_store.get(input_hash, [])
        
        if exclude_self:
            packets = [p for p in packets if p.agent_id != agent_id]
        
        # 更新统计
        self.agent_statistics[agent_id]['packets_received'] += len(packets)
        
        return packets
    
    def get_global_consensus(self, input_hash: str) -> Optional[SoftLabel]:
        """获取全局共识软标签"""
        packets = self.knowledge_store.get(input_hash, [])
        
        if not packets:
            return None
        
        # 简单平均聚合
        aggregated_logits: Dict[int, float] = defaultdict(float)
        
        for packet in packets:
            for class_id, logit in packet.soft_label.logits.items():
                aggregated_logits[class_id] += logit
        
        # 归一化
        n = len(packets)
        aggregated_logits = {k: v / n for k, v in aggregated_logits.items()}
        
        return SoftLabel(logits=aggregated_logits, temperature=2.0)
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """获取服务器统计"""
        return {
            'registered_agents': len(self.registered_agents),
            'total_knowledge_packets': sum(len(p) for p in self.knowledge_store.values()),
            'unique_inputs': len(self.knowledge_store),
            'agent_statistics': self.agent_statistics.copy()
        }


class ConsensusBasedDistillation:
    """基于共识的知识蒸馏"""
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.consensus_history: Dict[str, List[SoftLabel]] = defaultdict(list)
        self.consensus_weights: Dict[str, float] = {}
        
    def compute_consensus(self, packets: List[KnowledgePacket],
                         method: str = 'weighted_average') -> SoftLabel:
        """
        计算共识软标签
        
        Args:
            packets: 知识包列表
            method: 共识方法 ('average', 'weighted_average', 'entropy_weighted')
        """
        if not packets:
            return SoftLabel(logits={i: 0.0 for i in range(self.num_classes)})
        
        if method == 'average':
            return self._average_consensus(packets)
        elif method == 'weighted_average':
            return self._weighted_consensus(packets)
        elif method == 'entropy_weighted':
            return self._entropy_weighted_consensus(packets)
        else:
            return self._average_consensus(packets)
    
    def _average_consensus(self, packets: List[KnowledgePacket]) -> SoftLabel:
        """简单平均共识"""
        aggregated_logits: Dict[int, float] = defaultdict(float)
        
        for packet in packets:
            for class_id, logit in packet.soft_label.logits.items():
                aggregated_logits[class_id] += logit
        
        n = len(packets)
        aggregated_logits = {k: v / n for k, v in aggregated_logits.items()}
        
        return SoftLabel(logits=aggregated_logits)
    
    def _weighted_consensus(self, packets: List[KnowledgePacket]) -> SoftLabel:
        """加权平均共识(基于置信度)"""
        aggregated_logits: Dict[int, float] = defaultdict(float)
        total_weight = 0.0
        
        for packet in packets:
            weight = packet.confidence
            total_weight += weight
            
            for class_id, logit in packet.soft_label.logits.items():
                aggregated_logits[class_id] += weight * logit
        
        if total_weight > 0:
            aggregated_logits = {k: v / total_weight for k, v in aggregated_logits.items()}
        
        return SoftLabel(logits=aggregated_logits)
    
    def _entropy_weighted_consensus(self, packets: List[KnowledgePacket]) -> SoftLabel:
        """基于熵的加权共识(低熵=高置信度=高权重)"""
        aggregated_logits: Dict[int, float] = defaultdict(float)
        total_weight = 0.0
        
        for packet in packets:
            # 熵越低，权重越高
            entropy = packet.soft_label.get_entropy()
            max_entropy = math.log(self.num_classes)
            weight = 1.0 + (max_entropy - entropy)  # 范围 [1, 1+max_entropy]
            
            total_weight += weight
            
            for class_id, logit in packet.soft_label.logits.items():
                aggregated_logits[class_id] += weight * logit
        
        if total_weight > 0:
            aggregated_logits = {k: v / total_weight for k, v in aggregated_logits.items()}
        
        return SoftLabel(logits=aggregated_logits)
    
    def evaluate_consensus_quality(self, consensus: SoftLabel, 
                                   packets: List[KnowledgePacket]) -> float:
        """评估共识质量"""
        if not packets:
            return 0.0
        
        consensus_probs = consensus.get_probabilities()
        
        # 计算平均KL散度
        total_kl = 0.0
        for packet in packets:
            packet_probs = packet.soft_label.get_probabilities()
            kl = 0.0
            for k, p_consensus in consensus_probs.items():
                p_packet = packet_probs.get(k, 1e-10)
                if p_consensus > 0:
                    kl += p_consensus * math.log(p_consensus / max(p_packet, 1e-10))
            total_kl += kl
        
        avg_kl = total_kl / len(packets)
        # 质量 = 1 / (1 + KL)
        quality = 1.0 / (1.0 + avg_kl)
        
        return quality


class PersonalizedDistillation:
    """个性化知识蒸馏 - 根据Agent特性定制蒸馏"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.personalization_weights: Dict[str, float] = {}
        self.learning_style: str = 'balanced'  # 'conservative', 'balanced', 'aggressive'
        self.temperature_schedule: List[float] = [4.0, 2.0, 1.0]
        
    def set_learning_style(self, style: str):
        """设置学习风格"""
        self.learning_style = style
        
        if style == 'conservative':
            self.temperature_schedule = [2.0, 1.5, 1.0]
        elif style == 'aggressive':
            self.temperature_schedule = [8.0, 4.0, 1.0]
        else:  # balanced
            self.temperature_schedule = [4.0, 2.0, 1.0]
    
    def compute_personalized_weight(self, source_agent: str, 
                                    source_packet: KnowledgePacket) -> float:
        """计算个性化权重"""
        base_weight = self.personalization_weights.get(source_agent, 1.0)
        
        # 根据学习风格调整
        if self.learning_style == 'conservative':
            # 更信任高置信度的知识
            base_weight *= source_packet.confidence ** 2
        elif self.learning_style == 'aggressive':
            # 更积极探索不同知识
            base_weight *= (2.0 - source_packet.confidence)
        
        return base_weight
    
    def get_temperature(self, training_progress: float) -> float:
        """根据训练进度获取温度"""
        # training_progress: 0.0 - 1.0
        index = int(training_progress * len(self.temperature_schedule))
        index = min(index, len(self.temperature_schedule) - 1)
        return self.temperature_schedule[index]


class SecureKnowledgeExchange:
    """安全知识交换 - 保护知识交换隐私"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.access_control: Dict[str, Set[str]] = defaultdict(set)
        
    def _generate_key(self) -> str:
        """生成密钥"""
        return hashlib.sha256(str(random.random()).encode()).hexdigest()[:32]
    
    def grant_access(self, agent_id: str, allowed_recipients: List[str]):
        """授予访问权限"""
        self.access_control[agent_id] = set(allowed_recipients)
    
    def can_exchange(self, sender: str, receiver: str) -> bool:
        """检查是否可以交换"""
        allowed = self.access_control.get(sender, set())
        return receiver in allowed or len(allowed) == 0
    
    def anonymize_packet(self, packet: KnowledgePacket) -> KnowledgePacket:
        """匿名化知识包"""
        # 创建匿名副本
        return KnowledgePacket(
            packet_id=packet.packet_id,
            agent_id="anonymous",  # 隐藏真实身份
            input_hash=packet.input_hash,
            soft_label=packet.soft_label,
            timestamp=packet.timestamp,
            confidence=packet.confidence,
            version=packet.version,
            tags=packet.tags | {'anonymized'},
            metadata={k: v for k, v in packet.metadata.items() 
                     if k not in ['sensitive']}
        )
    
    def add_differential_privacy(self, soft_label: SoftLabel, 
                                 epsilon: float = 1.0) -> SoftLabel:
        """添加差分隐私噪声"""
        noisy_logits = {}
        for class_id, logit in soft_label.logits.items():
            # 添加拉普拉斯噪声
            noise = random.expovariate(epsilon) * (1 if random.random() > 0.5 else -1)
            noisy_logits[class_id] = logit + noise
        
        return SoftLabel(logits=noisy_logits, temperature=soft_label.temperature)
