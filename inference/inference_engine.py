"""
推理引擎模块 - Inference Engine
纯 Python 实现的综合推理引擎，涵盖文本生成、KV 缓存、动态批处理、
推测解码、量化推理、模型服务、性能基准测试和流式推理。
"""

import math
import time
import random
import hashlib
import threading
import heapq
import statistics
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Tuple, Any, Callable, Union, Iterator
)
from collections import defaultdict, deque, OrderedDict
from enum import Enum, auto


# ==================== 推理配置 ====================

@dataclass
class InferenceConfig:
    """推理配置，控制生成行为与资源限制"""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    num_beams: int = 1
    beam_length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    do_sample: bool = True
    contrastive_search_top_k: int = 4
    contrastive_search_penalty_alpha: float = 0.6
    eos_token_id: int = -1
    pad_token_id: int = 0
    seed: Optional[int] = None


# ==================== 文本生成器 ====================

class TextGenerator:
    """文本生成器 —— 实现多种解码策略"""

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()

    # ---------- 采样工具 ----------

    @staticmethod
    def _apply_temperature(logits: List[float], temperature: float) -> List[float]:
        if temperature <= 0:
            return logits
        inv_t = 1.0 / temperature
        return [l * inv_t for l in logits]

    @staticmethod
    def _apply_repetition_penalty(
        logits: List[float],
        token_ids: List[int],
        penalty: float,
    ) -> List[float]:
        if penalty == 1.0 or not token_ids:
            return logits
        result = list(logits)
        for tid in set(token_ids):
            if 0 <= tid < len(result):
                if result[tid] > 0:
                    result[tid] /= penalty
                else:
                    result[tid] *= penalty
        return result

    @staticmethod
    def _apply_frequency_presence_penalty(
        logits: List[float],
        token_counts: Dict[int, int],
        freq_penalty: float,
        pres_penalty: float,
    ) -> List[float]:
        if freq_penalty == 0.0 and pres_penalty == 0.0:
            return logits
        result = list(logits)
        for tid, cnt in token_counts.items():
            if 0 <= tid < len(result):
                result[tid] -= freq_penalty * cnt + pres_penalty * (1 if cnt > 0 else 0)
        return result

    @staticmethod
    def _softmax(logits: List[float]) -> List[float]:
        max_l = max(logits) if logits else 0.0
        exps = [math.exp(l - max_l) for l in logits]
        s = sum(exps)
        return [e / s for e in exps]

    @staticmethod
    def _log_softmax(logits: List[float]) -> List[float]:
        max_l = max(logits) if logits else 0.0
        lse = math.log(sum(math.exp(l - max_l) for l in logits)) + max_l
        return [l - lse for l in logits]

    @staticmethod
    def _top_k_filter(logits: List[float], k: int) -> List[float]:
        if k <= 0 or k >= len(logits):
            return logits
        threshold = sorted(logits, reverse=True)[k - 1]
        return [l if l >= threshold else float("-inf") for l in logits]

    @staticmethod
    def _top_p_filter(logits: List[float], p: float) -> List[float]:
        if p >= 1.0:
            return logits
        indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
        cumsum = 0.0
        keep = set()
        for idx, val in indexed:
            keep.add(idx)
            cumsum += math.exp(val)
            if cumsum >= p:
                break
        return [l if i in keep else float("-inf") for i, l in enumerate(logits)]

    @staticmethod
    def _sample_from_probs(probs: List[float], rng: random.Random) -> int:
        r = rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                return i
        return len(probs) - 1

    # ---------- 主生成接口 ----------

    def generate(
        self,
        logits_fn: Callable[[List[int]], List[float]],
        prompt_tokens: List[int],
        config: Optional[InferenceConfig] = None,
    ) -> List[int]:
        """根据 logits_fn 生成 token 序列。logits_fn(token_ids) -> logits"""
        cfg = config or self.config
        rng = random.Random(cfg.seed)
        tokens = list(prompt_tokens)
        token_counts: Dict[int, int] = defaultdict(int)
        for t in tokens:
            token_counts[t] += 1

        if cfg.num_beams > 1:
            return self._beam_search(logits_fn, tokens, cfg)

        for _ in range(cfg.max_new_tokens):
            logits = logits_fn(tokens)
            logits = self._apply_temperature(logits, cfg.temperature)
            logits = self._apply_repetition_penalty(logits, tokens, cfg.repetition_penalty)
            logits = self._apply_frequency_presence_penalty(
                logits, token_counts, cfg.frequency_penalty, cfg.presence_penalty
            )
            if cfg.do_sample:
                logits = self._top_k_filter(logits, cfg.top_k)
                logits = self._top_p_filter(logits, cfg.top_p)
                probs = self._softmax(logits)
                next_id = self._sample_from_probs(probs, rng)
            else:
                next_id = max(range(len(logits)), key=lambda i: logits[i])
            tokens.append(next_id)
            token_counts[next_id] += 1
            if next_id == cfg.eos_token_id:
                break
        return tokens

    # ---------- 束搜索 ----------

    def _beam_search(
        self,
        logits_fn: Callable[[List[int]], List[float]],
        prompt_tokens: List[int],
        cfg: InferenceConfig,
    ) -> List[int]:
        num_beams = cfg.num_beams
        beams: List[Tuple[float, List[int]]] = [(0.0, list(prompt_tokens))]
        finished: List[Tuple[float, List[int]]] = []

        for step in range(cfg.max_new_tokens):
            all_candidates: List[Tuple[float, List[int]]] = []
            for score, seq in beams:
                logits = logits_fn(seq)
                log_probs = self._log_softmax(logits)
                for tid, lp in enumerate(log_probs):
                    if tid == cfg.eos_token_id and step > 0:
                        new_score = score + lp
                        length = len(seq)
                        lp_penalty = ((5.0 + length) / (5.0 + 1.0)) ** cfg.beam_length_penalty
                        finished.append((new_score / lp_penalty, list(seq)))
                    else:
                        new_seq = seq + [tid]
                        new_score = score + lp
                        all_candidates.append((new_score, new_seq))

            if not all_candidates:
                break
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:num_beams]

            if len(finished) >= num_beams:
                break

        candidates = finished + [(s, seq) for s, seq in beams]
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1] if candidates else list(prompt_tokens)

    # ---------- 对比搜索 ----------

    def contrastive_search(
        self,
        logits_fn: Callable[[List[int]], List[float]],
        hidden_fn: Callable[[List[int]], List[List[float]]],
        prompt_tokens: List[int],
        top_k: int = 4,
        penalty_alpha: float = 0.6,
        max_new_tokens: int = 256,
    ) -> List[int]:
        """对比搜索：在 top-k 候选中选择与已有隐藏状态最相似的 token"""
        tokens = list(prompt_tokens)
        for _ in range(max_new_tokens):
            logits = logits_fn(tokens)
            probs = self._softmax(logits)
            top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
            last_hidden = hidden_fn(tokens)[-1]

            best_token = top_indices[0]
            best_score = float("-inf")
            for tid in top_indices:
                candidate_tokens = tokens + [tid]
                candidate_hidden = hidden_fn(candidate_tokens)[-1]
                sim = sum(a * b for a, b in zip(last_hidden, candidate_hidden))
                degenerated_penalty = penalty_alpha * sim
                score = probs[tid] - degenerated_penalty
                if score > best_score:
                    best_score = score
                    best_token = tid
            tokens.append(best_token)
        return tokens


# ==================== KV 缓存 ====================

class KVCache:
    """键值缓存，支持前缀缓存、分页块管理和缓存压缩"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        block_size: int = 16,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self._layers: List[Dict[int, List[float]]] = [
            {"keys": [], "values": []} for _ in range(num_layers)
        ]
        self._seq_len = 0
        # 前缀缓存：prefix_hash -> (seq_len, cached_layers_data)
        self._prefix_cache: Dict[str, Tuple[int, List]] = {}
        # 分页块管理
        self._num_blocks = max_seq_len // block_size
        self._free_blocks: List[int] = list(range(self._num_blocks))
        self._block_tables: Dict[int, List[int]] = {}  # req_id -> block ids
        self._block_data: Dict[int, List[List[float]]] = {}  # block_id -> data

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(self, layer_idx: int, new_keys: List[List[float]], new_values: List[List[float]]):
        layer = self._layers[layer_idx]
        layer["keys"].extend(new_keys)
        layer["values"].extend(new_values)
        self._seq_len = len(layer["keys"])

    def get(self, layer_idx: int) -> Tuple[List[List[float]], List[List[float]]]:
        layer = self._layers[layer_idx]
        return layer["keys"], layer["values"]

    def clear(self):
        for layer in self._layers:
            layer["keys"] = []
            layer["values"] = []
        self._seq_len = 0

    def trim(self, length: int):
        for layer in self._layers:
            layer["keys"] = layer["keys"][:length]
            layer["values"] = layer["values"][:length]
        self._seq_len = min(length, self._seq_len)

    # ---------- 前缀缓存 ----------

    def compute_prefix_hash(self, token_ids: List[int]) -> str:
        return hashlib.md5(",".join(map(str, token_ids[:64])).encode()).hexdigest()

    def store_prefix(self, token_ids: List[int]):
        h = self.compute_prefix_hash(token_ids)
        cached = [(list(layer["keys"]), list(layer["values"])) for layer in self._layers]
        self._prefix_cache[h] = (len(token_ids), cached)

    def load_prefix(self, token_ids: List[int]) -> bool:
        h = self.compute_prefix_hash(token_ids)
        if h not in self._prefix_cache:
            return False
        seq_len, cached = self._prefix_cache[h]
        for i, (keys, values) in enumerate(cached):
            self._layers[i]["keys"] = list(keys)
            self._layers[i]["values"] = list(values)
        self._seq_len = seq_len
        return True

    def evict_prefix(self, token_ids: List[int]):
        h = self.compute_prefix_hash(token_ids)
        self._prefix_cache.pop(h, None)

    # ---------- 分页块管理 (PagedAttention 风格) ----------

    def allocate_block(self, req_id: int) -> Optional[int]:
        if not self._free_blocks:
            return self._evict_lru_block()
        block_id = self._free_blocks.pop(0)
        if req_id not in self._block_tables:
            self._block_tables[req_id] = []
        self._block_tables[req_id].append(block_id)
        self._block_data[block_id] = []
        return block_id

    def write_to_block(self, block_id: int, data: List[float]):
        if block_id in self._block_data:
            self._block_data[block_id].extend(data)

    def read_block(self, block_id: int) -> List[float]:
        return self._block_data.get(block_id, [])

    def get_block_table(self, req_id: int) -> List[int]:
        return self._block_tables.get(req_id, [])

    def free_request_blocks(self, req_id: int):
        for bid in self._block_tables.pop(req_id, []):
            self._block_data.pop(bid, None)
            self._free_blocks.append(bid)

    def _evict_lru_block(self) -> Optional[int]:
        if not self._block_tables:
            return None
        oldest_req = next(iter(self._block_tables))
        blocks = self._block_tables[oldest_req]
        if blocks:
            evicted = blocks.pop()
            self._block_data.pop(evicted, None)
            return evicted
        return None

    # ---------- 缓存压缩 (滑动窗口摘要) ----------

    def compress(self, summary_ratio: float = 0.5):
        """通过保留每隔 N 个 token 来压缩缓存"""
        for layer in self._layers:
            k, v = layer["keys"], layer["values"]
            n = len(k)
            step = max(1, int(1.0 / summary_ratio))
            layer["keys"] = k[::step]
            layer["values"] = v[::step]
        self._seq_len = len(self._layers[0]["keys"])


# ==================== 批处理管理器 ====================

class RequestPriority(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    URGENT = auto()


@dataclass
class InferenceRequest:
    request_id: str
    prompt_tokens: List[int]
    priority: RequestPriority = RequestPriority.NORMAL
    max_new_tokens: int = 128
    arrival_time: float = 0.0
    preemption_count: int = 0


@dataclass
class InferenceResponse:
    request_id: str
    output_tokens: List[int]
    total_tokens: int
    ttft: float = 0.0
    tpot: float = 0.0
    preemption_count: int = 0


class BatchManager:
    """动态批处理管理器：连续批处理、优先级调度、抢占与重计算"""

    def __init__(
        self,
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 4096,
        scheduling_policy: str = "priority",  # priority | fcfs | sjf
    ):
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.scheduling_policy = scheduling_policy
        self._waiting_queue: List[InferenceRequest] = []
        self._running: Dict[str, InferenceRequest] = {}
        self._completed: List[InferenceResponse] = []
        self._lock = threading.Lock()
        self._req_states: Dict[str, Dict] = {}  # request_id -> {tokens_generated, start_time, ...}
        self._priority_map = {
            RequestPriority.LOW: 0,
            RequestPriority.NORMAL: 1,
            RequestPriority.HIGH: 2,
            RequestPriority.URGENT: 3,
        }

    def submit(self, request: InferenceRequest):
        with self._lock:
            request.arrival_time = time.time()
            self._waiting_queue.append(request)

    def schedule_batch(self) -> List[InferenceRequest]:
        """迭代级调度：选择下一批要执行的请求"""
        with self._lock:
            if self.scheduling_policy == "priority":
                self._waiting_queue.sort(
                    key=lambda r: (-self._priority_map[r.priority], r.arrival_time)
                )
            elif self.scheduling_policy == "sjf":
                self._waiting_queue.sort(key=lambda r: r.max_new_tokens)

            batch = []
            total_tokens = 0
            remaining = list(self._waiting_queue)

            for req in remaining:
                if len(batch) >= self.max_batch_size:
                    break
                state = self._req_states.get(req.request_id, {})
                tokens_so_far = state.get("tokens_generated", 0)
                if total_tokens + req.max_new_tokens - tokens_so_far > self.max_tokens_per_batch:
                    continue
                batch.append(req)
                self._running[req.request_id] = req
                total_tokens += req.max_new_tokens - tokens_so_far
                if req.request_id not in self._req_states:
                    self._req_states[req.request_id] = {
                        "tokens_generated": 0,
                        "start_time": time.time(),
                    }

            for req in batch:
                self._waiting_queue.remove(req)
            return batch

    def preempt(self, request_id: str):
        """抢占请求：将其移回等待队列"""
        with self._lock:
            req = self._running.pop(request_id, None)
            if req:
                req.preemption_count += 1
                req.priority = RequestPriority.HIGH
                self._waiting_queue.append(req)

    def update_progress(self, request_id: str, tokens_generated: int):
        with self._lock:
            if request_id in self._req_states:
                self._req_states[request_id]["tokens_generated"] = tokens_generated

    def complete_request(self, request_id: str, output_tokens: List[int]) -> InferenceResponse:
        with self._lock:
            req = self._running.pop(request_id, None)
            state = self._req_states.pop(request_id, {})
            start = state.get("start_time", time.time())
            ttft = state.get("first_token_time", start) - start
            elapsed = time.time() - start
            tpot = elapsed / max(1, len(output_tokens) - len(req.prompt_tokens)) if req else 0
            resp = InferenceResponse(
                request_id=request_id,
                output_tokens=output_tokens,
                total_tokens=len(output_tokens),
                ttft=ttft,
                tpot=tpot,
                preemption_count=req.preemption_count if req else 0,
            )
            self._completed.append(resp)
            return resp

    @property
    def waiting_count(self) -> int:
        return len(self._waiting_queue)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def completed_count(self) -> int:
        return len(self._completed)


# ==================== 推测解码 ====================

class SpeculativeDecoding:
    """推测解码：使用草稿模型加速目标模型推理"""

    def __init__(
        self,
        draft_logits_fn: Callable[[List[int]], List[float]],
        target_logits_fn: Callable[[List[int]], List[float]],
        num_draft_tokens: int = 5,
        temperature: float = 1.0,
    ):
        self.draft_fn = draft_logits_fn
        self.target_fn = target_logits_fn
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self._rng = random.Random(42)
        self._accepted_count = 0
        self._total_draft_count = 0

    @staticmethod
    def _softmax(logits: List[float]) -> List[float]:
        max_l = max(logits) if logits else 0.0
        exps = [math.exp(l - max_l) for l in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def _sample(self, probs: List[float]) -> int:
        r = self._rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                return i
        return len(probs) - 1

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 256,
        eos_id: int = -1,
    ) -> List[int]:
        """推测解码主循环"""
        tokens = list(prompt_tokens)
        while len(tokens) - len(prompt_tokens) < max_new_tokens:
            # 阶段 1：草稿模型生成 num_draft_tokens 个候选
            draft_tokens: List[int] = []
            draft_probs: List[List[float]] = []
            current = list(tokens)
            for _ in range(self.num_draft_tokens):
                logits = self.draft_fn(current)
                probs = self._softmax([l / self.temperature for l in logits])
                tok = self._sample(probs)
                draft_tokens.append(tok)
                draft_probs.append(probs)
                current.append(tok)
                if tok == eos_id:
                    break

            if not draft_tokens:
                break

            # 阶段 2：目标模型一次性验证所有候选
            verification_input = tokens + draft_tokens
            target_logits_all = self.target_fn(verification_input)
            vocab_size = len(target_logits_all) // len(verification_input)
            target_probs_all = [
                self._softmax(target_logits_all[i * vocab_size:(i + 1) * vocab_size])
                for i in range(len(verification_input))
            ]

            # 阶段 3：逐 token 验证（拒绝采样）
            n_accepted = 0
            for i, (dtok, dprob) in enumerate(zip(draft_tokens, draft_probs)):
                self._total_draft_count += 1
                t_idx = len(tokens) + i - 1
                if t_idx < 0:
                    t_idx = 0
                tprob = target_probs_all[min(t_idx, len(target_probs_all) - 1)]

                # 接受概率 = min(1, p_target(t) / p_draft(t))
                p_draft_t = dprob[dtok] if dtok < len(dprob) else 0.0
                p_target_t = tprob[dtok] if dtok < len(tprob) else 0.0
                accept_prob = min(1.0, p_target_t / (p_draft_t + 1e-30))

                if self._rng.random() < accept_prob:
                    tokens.append(dtok)
                    n_accepted += 1
                    self._accepted_count += 1
                    if dtok == eos_id:
                        return tokens
                else:
                    # 拒绝：从调整后的分布中采样
                    adjusted = [max(0.0, t - d) for t, d in zip(tprob, dprob)]
                    total = sum(adjusted)
                    if total > 0:
                        adjusted = [a / total for a in adjusted]
                    else:
                        adjusted = tprob
                    new_tok = self._sample(adjusted)
                    tokens.append(new_tok)
                    if new_tok == eos_id:
                        return tokens
                    break

            # 全部接受：从目标模型再采一个
            if n_accepted == len(draft_tokens):
                last_probs = target_probs_all[-1]
                new_tok = self._sample(last_probs)
                tokens.append(new_tok)
                if new_tok == eos_id:
                    return tokens

        return tokens

    def multi_draft_generate(
        self,
        draft_fns: List[Callable[[List[int]], List[float]]],
        target_fn: Callable[[List[int]], List[float]],
        prompt_tokens: List[int],
        max_new_tokens: int = 256,
        eos_id: int = -1,
    ) -> List[int]:
        """多草稿推测：使用多个草稿模型并行生成候选"""
        tokens = list(prompt_tokens)
        while len(tokens) - len(prompt_tokens) < max_new_tokens:
            all_drafts: List[List[int]] = []
            all_probs: List[List[List[float]]] = []

            for dfn in draft_fns:
                draft_tokens = []
                draft_probs = []
                current = list(tokens)
                for _ in range(self.num_draft_tokens):
                    logits = dfn(current)
                    probs = self._softmax([l / self.temperature for l in logits])
                    tok = self._sample(probs)
                    draft_tokens.append(tok)
                    draft_probs.append(probs)
                    current.append(tok)
                    if tok == eos_id:
                        break
                all_drafts.append(draft_tokens)
                all_probs.append(draft_probs)

            # 合并所有草稿候选并去重
            merged = []
            seen_positions = set()
            for d_tokens in all_drafts:
                for i, t in enumerate(d_tokens):
                    if i not in seen_positions:
                        merged.append(t)
                        seen_positions.add(i)

            if not merged:
                break

            verification_input = tokens + merged
            target_logits_all = target_fn(verification_input)
            vocab_size = len(target_logits_all) // len(verification_input)
            target_probs_all = [
                self._softmax(target_logits_all[i * vocab_size:(i + 1) * vocab_size])
                for i in range(len(verification_input))
            ]

            n_accepted = 0
            for i, dtok in enumerate(merged):
                t_idx = len(tokens) + i - 1
                if t_idx < 0:
                    t_idx = 0
                tprob = target_probs_all[min(t_idx, len(target_probs_all) - 1)]
                new_tok = max(range(len(tprob)), key=lambda j: tprob[j])
                if new_tok == dtok:
                    tokens.append(dtok)
                    n_accepted += 1
                    if dtok == eos_id:
                        return tokens
                else:
                    tokens.append(new_tok)
                    if new_tok == eos_id:
                        return tokens
                    break

            if n_accepted == len(merged):
                last_probs = target_probs_all[-1]
                new_tok = self._sample(last_probs)
                tokens.append(new_tok)
                if new_tok == eos_id:
                    return tokens

        return tokens

    @property
    def acceptance_rate(self) -> float:
        if self._total_draft_count == 0:
            return 0.0
        return self._accepted_count / self._total_draft_count


# ==================== 量化推理模拟 ====================

class QuantizedInference:
    """量化推理模拟：INT8/INT4、混合精度、仅权重量化"""

    def __init__(self):
        self._weights: Dict[str, List[float]] = {}
        self._scales: Dict[str, float] = {}
        self._zeros: Dict[str, int] = {}

    def register_weight(self, name: str, values: List[float]):
        self._weights[name] = values

    def quantize_weight(self, name: str, bits: int = 8) -> Tuple[List[int], float, int]:
        """将浮点权重量化为整数"""
        values = self._weights.get(name, [])
        if not values:
            return [], 1.0, 0
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        if range_val == 0:
            return [0] * len(values), 1.0, 0
        qmax = 2 ** (bits - 1) - 1
        qmin = -(2 ** (bits - 1))
        scale = range_val / (qmax - qmin)
        zero_point = round(-min_val / scale)
        quantized = [
            max(qmin, min(qmax, round(v / scale) + zero_point)) for v in values
        ]
        self._scales[name] = scale
        self._zeros[name] = zero_point
        return quantized, scale, zero_point

    def dequantize(self, name: str, quantized: List[int]) -> List[float]:
        """反量化"""
        scale = self._scales.get(name, 1.0)
        zero = self._zeros.get(name, 0)
        return [(q - zero) * scale for q in quantized]

    def quantize_linear(
        self, name: str, bits: int = 8
    ) -> Dict[str, Any]:
        """量化线性层权重并返回元数据"""
        q, scale, zp = self.quantize_weight(name, bits)
        return {"name": name, "bits": bits, "quantized": q, "scale": scale, "zero_point": zp}

    def mixed_precision_quantize(
        self, sensitivity_scores: Dict[str, float], threshold: float = 0.5
    ) -> Dict[str, int]:
        """混合精度量化：敏感层保持高精度，其余低精度"""
        plan = {}
        for name, score in sensitivity_scores.items():
            if score > threshold:
                plan[name] = 8  # INT8
            else:
                plan[name] = 4  # INT4
        return plan

    def weight_only_quantize(self, name: str, bits: int = 4) -> Dict[str, Any]:
        """仅权重量化（激活保持 FP16/FP32）"""
        q, scale, zp = self.quantize_weight(name, bits)
        return {
            "name": name,
            "bits": bits,
            "quantized_weights": q,
            "scale": scale,
            "zero_point": zp,
            "activation_dtype": "float32",
        }

    def simulated_matmul(
        self,
        quantized_weights: List[int],
        input_vec: List[float],
        scale: float,
        zero_point: int,
    ) -> List[float]:
        """模拟量化矩阵乘法"""
        out_dim = len(quantized_weights) // len(input_vec)
        result = []
        for row in range(out_dim):
            acc = 0.0
            for col in range(len(input_vec)):
                idx = row * len(input_vec) + col
                w_deq = (quantized_weights[idx] - zero_point) * scale
                acc += w_deq * input_vec[col]
            result.append(acc)
        return result

    def compute_quantization_error(
        self, name: str, quantized: List[int]
    ) -> Dict[str, float]:
        """计算量化误差指标"""
        original = self._weights.get(name, [])
        dequantized = self.dequantize(name, quantized)
        if not original or len(original) != len(dequantized):
            return {"mse": 0.0, "max_abs_error": 0.0, "mean_abs_error": 0.0}
        errors = [o - d for o, d in zip(original, dequantized)]
        mse = sum(e * e for e in errors) / len(errors)
        mae = sum(abs(e) for e in errors) / len(errors)
        max_ae = max(abs(e) for e in errors)
        return {"mse": mse, "max_abs_error": max_ae, "mean_abs_error": mae}


# ==================== 模型服务器 ====================

class ModelServer:
    """模型服务模拟：请求排队、负载均衡、速率限制、健康检查"""

    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 1000,
        rate_limit_rpm: int = 60,
        health_check_interval: float = 30.0,
    ):
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.rate_limit_rpm = rate_limit_rpm
        self.health_check_interval = health_check_interval

        self._request_queue: deque = deque(maxlen=max_queue_size)
        self._workers: List[Dict[str, Any]] = [
            {"id": i, "busy": False, "current_req": None, "processed": 0}
            for i in range(num_workers)
        ]
        self._request_timestamps: deque = deque()
        self._lock = threading.Lock()
        self._is_healthy = True
        self._total_served = 0
        self._total_rejected = 0
        self._start_time = time.time()
        self._health_check_thread: Optional[threading.Thread] = None

    def start(self):
        self._is_healthy = True
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()

    def stop(self):
        self._is_healthy = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)

    def submit_request(self, request_id: str, payload: Any = None) -> bool:
        """提交请求，返回是否被接受"""
        with self._lock:
            if not self._is_healthy:
                self._total_rejected += 1
                return False
            if not self._check_rate_limit():
                self._total_rejected += 1
                return False
            if len(self._request_queue) >= self.max_queue_size:
                self._total_rejected += 1
                return False
            self._request_timestamps.append(time.time())
            self._request_queue.append({"id": request_id, "payload": payload, "submit_time": time.time()})
            return True

    def _check_rate_limit(self) -> bool:
        now = time.time()
        window = 60.0
        while self._request_timestamps and self._request_timestamps[0] < now - window:
            self._request_timestamps.popleft()
        return len(self._request_timestamps) < self.rate_limit_rpm

    def assign_requests(self) -> List[Dict[str, Any]]:
        """负载均衡：将排队请求分配给空闲 worker"""
        with self._lock:
            assignments = []
            for worker in self._workers:
                if not worker["busy"] and self._request_queue:
                    req = self._request_queue.popleft()
                    worker["busy"] = True
                    worker["current_req"] = req["id"]
                    assignments.append(req)
            return assignments

    def complete_request(self, request_id: str):
        with self._lock:
            for worker in self._workers:
                if worker["current_req"] == request_id:
                    worker["busy"] = False
                    worker["current_req"] = None
                    worker["processed"] += 1
                    self._total_served += 1
                    break

    def _health_check_loop(self):
        while self._is_healthy:
            time.sleep(self.health_check_interval)
            busy_count = sum(1 for w in self._workers if w["busy"])
            queue_len = len(self._request_queue)
            if busy_count == self.num_workers and queue_len > self.max_queue_size * 0.8:
                self._is_healthy = False

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    @property
    def queue_size(self) -> int:
        return len(self._request_queue)

    @property
    def stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        return {
            "total_served": self._total_served,
            "total_rejected": self._total_rejected,
            "queue_size": len(self._request_queue),
            "workers_busy": sum(1 for w in self._workers if w["busy"]),
            "workers_total": self.num_workers,
            "throughput_rps": self._total_served / max(elapsed, 0.001),
            "uptime_seconds": elapsed,
            "is_healthy": self._is_healthy,
        }


# ==================== 性能基准测试 ====================

@dataclass
class BenchmarkResult:
    name: str
    throughput_tokens_per_sec: float
    ttft_ms: float  # Time To First Token
    tpot_ms: float  # Time Per Output Token
    tbt_ms: float   # Total Batch Time
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    memory_peak_mb: float
    total_requests: int
    total_tokens: int
    duration_seconds: float


class BenchmarkRunner:
    """性能基准测试：吞吐量、延迟、内存使用"""

    def __init__(self):
        self._latencies: List[float] = []
        self._ttfts: List[float] = []
        self._tpots: List[float] = []
        self._memory_samples: List[float] = []

    def run_benchmark(
        self,
        inference_fn: Callable[[List[int]], Tuple[List[int], float, float]],
        prompts: List[List[int]],
        num_warmup: int = 3,
    ) -> BenchmarkResult:
        """
        运行基准测试。
        inference_fn(prompt) -> (output_tokens, ttft_seconds, tpot_seconds)
        """
        self._latencies.clear()
        self._ttfts.clear()
        self._tpots.clear()
        self._memory_samples.clear()

        # 预热
        for prompt in prompts[:num_warmup]:
            inference_fn(prompt)

        total_tokens = 0
        start_time = time.time()

        for prompt in prompts[num_warmup:]:
            mem_before = self._estimate_memory()
            req_start = time.time()
            output, ttft, tpot = inference_fn(prompt)
            req_end = time.time()
            mem_after = self._estimate_memory()

            latency = (req_end - req_start) * 1000  # ms
            self._latencies.append(latency)
            self._ttfts.append(ttft * 1000)
            self._tpots.append(tpot * 1000)
            self._memory_samples.append(max(mem_before, mem_after))
            total_tokens += len(output)

        duration = time.time() - start_time
        total_requests = len(prompts) - num_warmup

        return BenchmarkResult(
            name="inference_benchmark",
            throughput_tokens_per_sec=total_tokens / max(duration, 0.001),
            ttft_ms=self._percentile(self._ttfts, 50),
            tpot_ms=self._percentile(self._tpots, 50),
            tbt_ms=duration * 1000 / max(total_requests, 1),
            latency_p50_ms=self._percentile(self._latencies, 50),
            latency_p90_ms=self._percentile(self._latencies, 90),
            latency_p99_ms=self._percentile(self._latencies, 99),
            memory_peak_mb=max(self._memory_samples) if self._memory_samples else 0.0,
            total_requests=total_requests,
            total_tokens=total_tokens,
            duration_seconds=duration,
        )

    def run_concurrent_benchmark(
        self,
        inference_fn: Callable[[List[int]], Tuple[List[int], float, float]],
        prompts: List[List[int]],
        concurrency: int = 4,
    ) -> BenchmarkResult:
        """并发基准测试"""
        results: List[BenchmarkResult] = []
        chunk_size = max(1, len(prompts) // concurrency)
        threads = []

        def worker(chunk):
            runner = BenchmarkRunner()
            return runner.run_benchmark(inference_fn, chunk, num_warmup=0)

        chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]
        for chunk in chunks:
            t = threading.Thread(target=lambda c=chunk: results.append(worker(c)))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        if not results:
            return BenchmarkResult(
                name="concurrent_benchmark", throughput_tokens_per_sec=0, ttft_ms=0,
                tpot_ms=0, tbt_ms=0, latency_p50_ms=0, latency_p90_ms=0,
                latency_p99_ms=0, memory_peak_mb=0, total_requests=0,
                total_tokens=0, duration_seconds=0,
            )

        total_tokens = sum(r.total_tokens for r in results)
        total_duration = max(r.duration_seconds for r in results)
        all_latencies = []
        for r in results:
            all_latencies.extend(self._latencies)

        return BenchmarkResult(
            name=f"concurrent_{concurrency}",
            throughput_tokens_per_sec=total_tokens / max(total_duration, 0.001),
            ttft_ms=statistics.median(self._ttfts) if self._ttfts else 0,
            tpot_ms=statistics.median(self._tpots) if self._tpots else 0,
            tbt_ms=total_duration * 1000 / max(len(prompts), 1),
            latency_p50_ms=self._percentile(all_latencies, 50) if all_latencies else 0,
            latency_p90_ms=self._percentile(all_latencies, 90) if all_latencies else 0,
            latency_p99_ms=self._percentile(all_latencies, 99) if all_latencies else 0,
            memory_peak_mb=max((r.memory_peak_mb for r in results), default=0),
            total_requests=len(prompts),
            total_tokens=total_tokens,
            duration_seconds=total_duration,
        )

    @staticmethod
    def _percentile(data: List[float], pct: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * pct / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

    @staticmethod
    def _estimate_memory() -> float:
        """模拟内存估算（返回 MB）"""
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except (ImportError, AttributeError):
            return 0.0

    def summary(self, result: BenchmarkResult) -> str:
        lines = [
            f"=== 基准测试结果: {result.name} ===",
            f"  吞吐量: {result.throughput_tokens_per_sec:.1f} tokens/sec",
            f"  TTFT (P50): {result.ttft_ms:.2f} ms",
            f"  TPOT (P50): {result.tpot_ms:.2f} ms",
            f"  TBT: {result.tbt_ms:.2f} ms",
            f"  延迟 P50/P90/P99: {result.latency_p50_ms:.1f}/{result.latency_p90_ms:.1f}/{result.latency_p99_ms:.1f} ms",
            f"  内存峰值: {result.memory_peak_mb:.1f} MB",
            f"  总请求: {result.total_requests}, 总 token: {result.total_tokens}",
            f"  持续时间: {result.duration_seconds:.2f} 秒",
        ]
        return "\n".join(lines)


# ==================== 流式推理 ====================

class StreamingInference:
    """流式推理：逐 token 流式输出、分块流式、SSE 模拟"""

    def __init__(
        self,
        logits_fn: Callable[[List[int]], List[float]],
        config: Optional[InferenceConfig] = None,
    ):
        self.logits_fn = logits_fn
        self.config = config or InferenceConfig()
        self._generator = TextGenerator(self.config)

    def stream_tokens(
        self,
        prompt_tokens: List[int],
    ) -> Iterator[Dict[str, Any]]:
        """逐 token 流式生成"""
        tokens = list(prompt_tokens)
        token_counts: Dict[int, int] = defaultdict(int)
        for t in tokens:
            token_counts[t] += 1
        rng = random.Random(self.config.seed)
        first_token_time = None

        for step in range(self.config.max_new_tokens):
            t0 = time.time()
            logits = self.logits_fn(tokens)
            logits = self._generator._apply_temperature(logits, self.config.temperature)
            logits = self._generator._apply_repetition_penalty(logits, tokens, self.config.repetition_penalty)
            logits = self._generator._apply_frequency_presence_penalty(
                logits, token_counts, self.config.frequency_penalty, self.config.presence_penalty
            )
            if self.config.do_sample:
                logits = self._generator._top_k_filter(logits, self.config.top_k)
                logits = self._generator._top_p_filter(logits, self.config.top_p)
                probs = self._generator._softmax(logits)
                next_id = self._generator._sample_from_probs(probs, rng)
            else:
                next_id = max(range(len(logits)), key=lambda i: logits[i])

            tokens.append(next_id)
            token_counts[next_id] += 1
            latency_ms = (time.time() - t0) * 1000

            if first_token_time is None:
                first_token_time = latency_ms

            yield {
                "token_id": next_id,
                "token_index": step,
                "latency_ms": latency_ms,
                "ttft_ms": first_token_time,
                "is_finished": next_id == self.config.eos_token_id or step == self.config.max_new_tokens - 1,
                "prompt_tokens": len(prompt_tokens),
                "generated_tokens": step + 1,
            }

            if next_id == self.config.eos_token_id:
                break

    def stream_chunks(
        self,
        prompt_tokens: List[int],
        chunk_size: int = 5,
    ) -> Iterator[Dict[str, Any]]:
        """分块流式输出"""
        chunk_tokens = []
        chunk_start = time.time()

        for event in self.stream_tokens(prompt_tokens):
            chunk_tokens.append(event["token_id"])
            if len(chunk_tokens) >= chunk_size or event["is_finished"]:
                yield {
                    "tokens": chunk_tokens,
                    "chunk_size": len(chunk_tokens),
                    "latency_ms": (time.time() - chunk_start) * 1000,
                    "is_finished": event["is_finished"],
                }
                chunk_tokens = []
                chunk_start = time.time()

    def stream_sse(
        self,
        prompt_tokens: List[int],
    ) -> Iterator[str]:
        """Server-Sent Events 格式流式输出"""
        for event in self.stream_tokens(prompt_tokens):
            sse_data = (
                f"data: {event}\n\n"
            )
            yield sse_data
            if event["is_finished"]:
                yield "data: [DONE]\n\n"


# ==================== 主函数 ====================

def _demo_logits_fn(tokens: List[int]) -> List[float]:
    """演示用 logits 函数：返回伪随机 logits"""
    vocab_size = 1000
    rng = random.Random(hash(tuple(tokens[-8:])) if tokens else 0)
    logits = [rng.gauss(0, 1) for _ in range(vocab_size)]
    # 偏向较低编号的 token 以模拟语言模型
    for i in range(min(50, vocab_size)):
        logits[i] += rng.uniform(0.5, 2.0)
    return logits


def _demo_hidden_fn(tokens: List[int]) -> List[List[float]]:
    """演示用隐藏状态函数"""
    dim = 64
    rng = random.Random(hash(tuple(tokens[-4:])) if tokens else 0)
    return [[rng.gauss(0, 0.1) for _ in range(dim)] for _ in range(len(tokens))]


def main():
    print("=" * 60)
    print("  推理引擎演示 (纯 Python)")
    print("=" * 60)

    # 1. 文本生成
    print("\n--- 文本生成 ---")
    config = InferenceConfig(max_new_tokens=20, temperature=0.8, top_k=10, top_p=0.9, seed=42)
    gen = TextGenerator(config)
    prompt = [1, 50, 23, 87]
    output = gen.generate(_demo_logits_fn, prompt)
    print(f"贪心/采样生成: prompt={prompt} -> output长度={len(output)}, tokens={output}")

    # 束搜索
    beam_cfg = InferenceConfig(num_beams=3, max_new_tokens=15, beam_length_penalty=1.0, seed=42)
    beam_gen = TextGenerator(beam_cfg)
    beam_output = beam_gen.generate(_demo_logits_fn, prompt)
    print(f"束搜索 (beams=3): output长度={len(beam_output)}, tokens={beam_output}")

    # 对比搜索
    contrastive_output = gen.contrastive_search(
        _demo_logits_fn, _demo_hidden_fn, prompt, top_k=4, penalty_alpha=0.6, max_new_tokens=10
    )
    print(f"对比搜索: output长度={len(contrastive_output)}, tokens={contrastive_output}")

    # 2. KV 缓存
    print("\n--- KV 缓存 ---")
    kv = KVCache(num_layers=4, num_heads=8, head_dim=64, max_seq_len=512, block_size=16)
    dummy_k = [[float(i)] * 64 for i in range(8)]
    dummy_v = [[float(i + 100)] * 64 for i in range(8)]
    kv.update(0, dummy_k, dummy_v)
    print(f"缓存更新后 seq_len={kv.seq_len}")
    kv.store_prefix([1, 50, 23])
    loaded = kv.load_prefix([1, 50, 23])
    print(f"前缀缓存加载: {loaded}")
    block_id = kv.allocate_block(req_id=1)
    print(f"分配块: block_id={block_id}, 空闲块数={len(kv._free_blocks)}")
    kv.free_request_blocks(1)
    print(f"释放后空闲块数={len(kv._free_blocks)}")
    kv.update(0, dummy_k, dummy_v)
    kv.update(0, dummy_k, dummy_v)
    kv.compress(0.5)
    print(f"压缩后 seq_len={kv.seq_len}")

    # 3. 批处理管理器
    print("\n--- 批处理管理器 ---")
    bm = BatchManager(max_batch_size=4, max_tokens_per_batch=1024, scheduling_policy="priority")
    for i in range(6):
        priority = RequestPriority.HIGH if i < 2 else RequestPriority.NORMAL
        bm.submit(InferenceRequest(f"req_{i}", [i], priority=priority, max_new_tokens=64))
    batch = bm.schedule_batch()
    print(f"调度批次大小={len(batch)}, 等待队列={bm.waiting_count}, 运行中={bm.running_count}")
    bm.complete_request(batch[0].request_id, [1, 2, 3])
    print(f"完成后: 运行中={bm.running_count}, 已完成={bm.completed_count}")

    # 4. 推测解码
    print("\n--- 推测解码 ---")
    sd = SpeculativeDecoding(_demo_logits_fn, _demo_logits_fn, num_draft_tokens=3, temperature=0.8)
    spec_output = sd.generate(prompt, max_new_tokens=15, eos_id=-1)
    print(f"推测解码: output长度={len(spec_output)}, 接受率={sd.acceptance_rate:.2%}")

    # 5. 量化推理
    print("\n--- 量化推理 ---")
    qi = QuantizedInference()
    weight = [random.uniform(-2, 2) for _ in range(128)]
    qi.register_weight("linear1", weight)
    q8, scale8, zp8 = qi.quantize_weight("linear1", bits=8)
    q4, scale4, zp4 = qi.quantize_weight("linear1", bits=4)
    err8 = qi.compute_quantization_error("linear1", q8)
    err4 = qi.compute_quantization_error("linear1", q4)
    print(f"INT8 量化: MSE={err8['mse']:.6f}, MAE={err8['mean_abs_error']:.6f}")
    print(f"INT4 量化: MSE={err4['mse']:.6f}, MAE={err4['mean_abs_error']:.6f}")
    sensitivity = {"layer1": 0.8, "layer2": 0.3, "layer3": 0.6}
    plan = qi.mixed_precision_quantize(sensitivity, threshold=0.5)
    print(f"混合精度方案: {plan}")

    # 6. 模型服务器
    print("\n--- 模型服务器 ---")
    server = ModelServer(num_workers=2, max_queue_size=10, rate_limit_rpm=1000)
    server.start()
    for i in range(5):
        accepted = server.submit_request(f"svc_{i}")
        print(f"  请求 svc_{i}: {'接受' if accepted else '拒绝'}")
    assignments = server.assign_requests()
    print(f"  分配请求: {[a['id'] for a in assignments]}")
    server.complete_request(assignments[0]["id"])
    print(f"  服务器统计: {server.stats}")
    server.stop()

    # 7. 性能基准测试
    print("\n--- 性能基准测试 ---")
    runner = BenchmarkRunner()

    def bench_fn(prompt):
        t0 = time.time()
        out = gen.generate(_demo_logits_fn, prompt, InferenceConfig(max_new_tokens=10, seed=42))
        elapsed = time.time() - t0
        return out, elapsed * 0.3, elapsed / max(len(out) - len(prompt), 1) * 0.7

    prompts = [[random.randint(0, 99) for _ in range(5)] for _ in range(8)]
    result = runner.run_benchmark(bench_fn, prompts, num_warmup=2)
    print(runner.summary(result))

    # 8. 流式推理
    print("\n--- 流式推理 ---")
    streamer = StreamingInference(_demo_logits_fn, InferenceConfig(max_new_tokens=8, temperature=0.8, seed=42))
    token_count = 0
    for event in streamer.stream_tokens([1, 2, 3]):
        token_count += 1
        if token_count <= 3:
            print(f"  Token {event['token_index']}: id={event['token_id']}, "
                  f"latency={event['latency_ms']:.2f}ms, ttft={event['ttft_ms']:.2f}ms")
    print(f"  共生成 {token_count} 个 token")

    chunk_count = 0
    for chunk in streamer.stream_chunks([1, 2, 3], chunk_size=3):
        chunk_count += 1
        print(f"  Chunk {chunk_count}: {len(chunk['tokens'])} tokens, "
              f"latency={chunk['latency_ms']:.2f}ms")

    print("\n" + "=" * 60)
    print("  推理引擎演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
