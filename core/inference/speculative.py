"""
投机解码真实LLM实现 - Speculative Decoding with Real LLM

集成 transformers 的 draft model
实现验证和接受逻辑

作者: UFO Framework Team
"""

import time
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
        TopKLogitsWarper,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available, using mock implementation")


class SpeculativeMode(Enum):
    """投机解码模式"""
    STANDARD = "standard"  # 标准投机解码
    MEDUSA = "medusa"      # Medusa多头预测
    EAGLE = "eagle"        # Eagle特征层融合
    LOOKAHEAD = "lookahead"  # Lookahead解码


@dataclass
class SpeculativeConfig:
    """投机解码配置"""
    # 模型配置
    target_model_name: str = "meta-llama/Llama-2-7b-hf"
    draft_model_name: Optional[str] = None  # 如果为None，使用目标模型作为草稿模型

    # 投机解码参数
    num_speculative_tokens: int = 5
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    # 验证参数
    min_acceptance_prob: float = 0.0  # 最小接受概率阈值
    adaptive_draft_length: bool = True  # 自适应草稿长度

    # 性能优化
    use_cache: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16

    # 调试
    verbose: bool = False


@dataclass
class SpeculativeStats:
    """投机解码统计信息"""
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0
    total_target_tokens: int = 0
    total_rounds: int = 0
    total_time: float = 0.0
    draft_time: float = 0.0
    verify_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted_tokens / max(self.total_draft_tokens, 1)

    @property
    def speedup_ratio(self) -> float:
        """加速比 = 无投机解码所需Token数 / 实际目标模型生成Token数"""
        if self.total_target_tokens == 0:
            return 1.0
        effective_tokens = self.total_accepted_tokens + self.total_target_tokens
        return effective_tokens / max(self.total_target_tokens, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_rejected_tokens": self.total_rejected_tokens,
            "total_target_tokens": self.total_target_tokens,
            "total_rounds": self.total_rounds,
            "total_time": round(self.total_time, 4),
            "draft_time": round(self.draft_time, 4),
            "verify_time": round(self.verify_time, 4),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "speedup_ratio": round(self.speedup_ratio, 4),
        }


class DraftModel:
    """
    草稿模型封装

    使用较小的模型快速生成候选token
    """

    def __init__(
        self,
        model_name: Optional[str],
        target_model: Optional[Any] = None,
        config: Optional[SpeculativeConfig] = None,
    ):
        self.config = config or SpeculativeConfig()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.is_target_model = model_name is None

        if TRANSFORMERS_AVAILABLE:
            self._load_model(model_name, target_model)
        else:
            self._init_mock()

    def _load_model(
        self,
        model_name: Optional[str],
        target_model: Optional[Any],
    ):
        """加载草稿模型"""
        if model_name is None and target_model is not None:
            # 使用目标模型作为草稿模型（自投机）
            self.model = target_model
            self.is_target_model = True
            if self.config.verbose:
                print("Using target model as draft model (self-speculation)")
        elif model_name:
            # 加载独立的草稿模型
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=self.config.dtype,
                    device_map=self.config.device,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.config.verbose:
                    print(f"Loaded draft model: {model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load draft model: {e}")
                self.model = None
        else:
            self.model = None

    def _init_mock(self):
        """初始化mock模型"""
        self.model = None
        self.tokenizer = None

    def generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[List[int], List[float]]:
        """
        生成草稿token序列

        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            num_tokens: 生成的token数量
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数

        Returns:
            (token_ids, log_probs)
        """
        if self.model is None:
            return self._mock_generate(input_ids, num_tokens)

        draft_tokens = []
        draft_log_probs = []
        current_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(num_tokens):
                # 前向传播
                outputs = self.model(current_ids, use_cache=self.config.use_cache)
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

                # 应用temperature
                if temperature != 1.0 and temperature > 0:
                    logits = logits / temperature

                # 应用top-p和top-k
                probs = F.softmax(logits, dim=-1)

                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                # 采样
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()
                log_prob = torch.log(probs[0, token_id]).item()

                draft_tokens.append(token_id)
                draft_log_probs.append(log_prob)

                # 更新输入
                current_ids = torch.cat([current_ids, next_token], dim=-1)

        return draft_tokens, draft_log_probs

    def _mock_generate(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> Tuple[List[int], List[float]]:
        """Mock生成（用于测试）"""
        import random
        vocab_size = 50000
        tokens = [random.randint(0, vocab_size - 1) for _ in range(num_tokens)]
        log_probs = [random.uniform(-5, -0.1) for _ in range(num_tokens)]
        return tokens, log_probs


class TargetModel:
    """
    目标模型封装

    使用大模型验证草稿token
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[SpeculativeConfig] = None,
    ):
        self.config = config or SpeculativeConfig()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        if TRANSFORMERS_AVAILABLE:
            self._load_model(model_name)
        else:
            self._init_mock()

    def _load_model(self, model_name: str):
        """加载目标模型"""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.config.dtype,
                device_map=self.config.device,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.config.verbose:
                print(f"Loaded target model: {model_name}")
        except Exception as e:
            warnings.warn(f"Failed to load target model: {e}")
            self.model = None

    def _init_mock(self):
        """初始化mock模型"""
        self.model = None
        self.tokenizer = None

    def verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: List[int],
        temperature: float = 1.0,
    ) -> Tuple[List[int], int, List[float]]:
        """
        验证草稿token

        Args:
            input_ids: 输入token IDs
            draft_tokens: 草稿模型生成的token列表
            temperature: 温度参数

        Returns:
            (accepted_tokens, first_rejected_idx, target_log_probs)
            first_rejected_idx: -1表示全部接受
        """
        if self.model is None:
            return self._mock_verify(input_ids, draft_tokens)

        accepted_tokens = []
        target_log_probs = []
        current_ids = input_ids.clone()

        with torch.no_grad():
            for i, draft_token in enumerate(draft_tokens):
                # 目标模型前向传播
                outputs = self.model(current_ids, use_cache=self.config.use_cache)
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

                # 应用temperature
                if temperature != 1.0 and temperature > 0:
                    logits = logits / temperature

                # 计算概率
                probs = F.softmax(logits, dim=-1)
                target_token = torch.argmax(probs, dim=-1).item()
                target_log_prob = torch.log(probs[0, draft_token]).item()
                target_log_probs.append(target_log_prob)

                # 验证：比较草稿token和目标模型token
                if draft_token == target_token:
                    # 接受token
                    accepted_tokens.append(draft_token)
                    current_ids = torch.cat([
                        current_ids,
                        torch.tensor([[draft_token]], device=current_ids.device)
                    ], dim=-1)
                else:
                    # 拒绝token，使用目标模型的token
                    accepted_tokens.append(target_token)
                    return accepted_tokens, i, target_log_probs

        # 全部接受
        return accepted_tokens, -1, target_log_probs

    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[int, float]:
        """
        生成下一个token

        Args:
            input_ids: 输入token IDs
            temperature: 温度参数

        Returns:
            (token_id, log_prob)
        """
        if self.model is None:
            import random
            return random.randint(0, 50000), -1.0

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=self.config.use_cache)
            logits = outputs.logits[:, -1, :]

            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            log_prob = torch.log(probs[0, token_id]).item()

        return token_id, log_prob

    def _mock_verify(
        self,
        input_ids: torch.Tensor,
        draft_tokens: List[int],
    ) -> Tuple[List[int], int, List[float]]:
        """Mock验证（用于测试）"""
        import random
        # 模拟70%接受率
        accepted = []
        for i, token in enumerate(draft_tokens):
            if random.random() < 0.7:
                accepted.append(token)
            else:
                accepted.append(random.randint(0, 50000))
                return accepted, i, [random.uniform(-5, -0.1) for _ in range(len(accepted))]
        return accepted, -1, [random.uniform(-5, -0.1) for _ in range(len(accepted))]


class SpeculativeDecoder:
    """
    投机解码器

    实现完整的投机解码流程：
    1. 草稿模型快速生成候选token
    2. 目标模型并行验证
    3. 接受/拒绝逻辑
    4. 从拒绝位置继续生成
    """

    def __init__(self, config: Optional[SpeculativeConfig] = None):
        self.config = config or SpeculativeConfig()
        self.stats = SpeculativeStats()

        # 初始化模型
        self.target_model = TargetModel(
            self.config.target_model_name,
            self.config,
        )
        self.draft_model = DraftModel(
            self.config.draft_model_name,
            self.target_model.model,
            self.config,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        投机解码生成

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            stop_sequences: 停止序列

        Returns:
            包含生成结果和统计信息的字典
        """
        if not TRANSFORMERS_AVAILABLE or self.target_model.tokenizer is None:
            return self._mock_generate(prompt, max_new_tokens)

        # 使用配置默认值
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k

        # 编码输入
        input_ids = self.target_model.tokenizer.encode(
            prompt,
            return_tensors="pt",
        ).to(self.config.device)

        generated_tokens = []
        start_time = time.time()

        while len(generated_tokens) < max_new_tokens:
            round_start = time.time()

            # 1. 草稿模型生成候选token
            draft_start = time.time()
            num_draft = self._get_draft_length()
            draft_tokens, draft_log_probs = self.draft_model.generate_draft_tokens(
                input_ids,
                num_tokens=num_draft,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            draft_time = time.time() - draft_start
            self.stats.draft_time += draft_time
            self.stats.total_draft_tokens += len(draft_tokens)

            # 2. 目标模型验证
            verify_start = time.time()
            accepted_tokens, rejected_idx, target_log_probs = self.target_model.verify_tokens(
                input_ids,
                draft_tokens,
                temperature=temperature,
            )
            verify_time = time.time() - verify_start
            self.stats.verify_time += verify_time

            # 3. 更新统计
            if rejected_idx == -1:
                # 全部接受
                self.stats.total_accepted_tokens += len(draft_tokens)
            else:
                # 部分接受
                self.stats.total_accepted_tokens += rejected_idx
                self.stats.total_rejected_tokens += len(draft_tokens) - rejected_idx

            self.stats.total_target_tokens += 1  # 至少调用一次目标模型
            self.stats.total_rounds += 1

            # 4. 更新生成结果
            generated_tokens.extend(accepted_tokens)

            # 5. 更新输入
            new_tokens = torch.tensor([accepted_tokens], device=input_ids.device)
            input_ids = torch.cat([input_ids, new_tokens], dim=-1)

            # 6. 检查停止条件
            if rejected_idx == -1 and len(accepted_tokens) < num_draft:
                # 草稿模型提前结束
                break

            # 检查是否达到最大长度
            if len(generated_tokens) >= max_new_tokens:
                break

            # 自适应调整草稿长度
            if self.config.adaptive_draft_length:
                self._adapt_draft_length(rejected_idx)

        self.stats.total_time = time.time() - start_time

        # 解码生成结果
        generated_text = self.target_model.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return {
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "prompt": prompt,
            "stats": self.stats.to_dict(),
        }

    def _get_draft_length(self) -> int:
        """获取当前草稿长度"""
        return self.config.num_speculative_tokens

    def _adapt_draft_length(self, rejected_idx: int):
        """自适应调整草稿长度"""
        # 简单策略：如果全部接受，增加草稿长度；如果被拒绝，减少草稿长度
        if rejected_idx == -1:
            # 全部接受，可以适当增加
            pass
        else:
            # 被拒绝，减少草稿长度
            self.config.num_speculative_tokens = max(
                1,
                self.config.num_speculative_tokens - 1
            )

    def _mock_generate(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        """Mock生成（用于测试）"""
        import random
        start_time = time.time()

        # 模拟生成
        generated_tokens = [random.randint(0, 50000) for _ in range(max_new_tokens)]

        self.stats.total_time = time.time() - start_time
        self.stats.total_draft_tokens = max_new_tokens
        self.stats.total_accepted_tokens = int(max_new_tokens * 0.7)
        self.stats.total_target_tokens = max_new_tokens - self.stats.total_accepted_tokens
        self.stats.total_rounds = max_new_tokens // self.config.num_speculative_tokens

        return {
            "generated_text": "[Mock generated text] " * 10,
            "generated_tokens": generated_tokens,
            "prompt": prompt,
            "stats": self.stats.to_dict(),
        }

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Iterator[str]:
        """
        流式生成

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        result = self.generate(prompt, max_new_tokens, **kwargs)

        # 模拟流式输出
        text = result["generated_text"]
        words = text.split()

        for word in words:
            yield word + " "

    def reset_stats(self):
        """重置统计信息"""
        self.stats = SpeculativeStats()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.to_dict()


def create_speculative_decoder(
    target_model: str,
    draft_model: Optional[str] = None,
    num_speculative_tokens: int = 5,
    **kwargs
) -> SpeculativeDecoder:
    """
    便捷函数：创建投机解码器

    Args:
        target_model: 目标模型名称或路径
        draft_model: 草稿模型名称或路径（None表示使用目标模型）
        num_speculative_tokens: 投机token数量
        **kwargs: 其他配置参数

    Returns:
        SpeculativeDecoder实例
    """
    config = SpeculativeConfig(
        target_model_name=target_model,
        draft_model_name=draft_model,
        num_speculative_tokens=num_speculative_tokens,
        **kwargs
    )
    return SpeculativeDecoder(config)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("投机解码真实LLM实现测试")
    print("=" * 60)

    # 创建投机解码器
    print("\n[1] 创建投机解码器")
    decoder = create_speculative_decoder(
        target_model="mock-model",  # 使用mock模式
        draft_model=None,
        num_speculative_tokens=5,
        verbose=True,
    )

    # 测试生成
    print("\n[2] 测试投机解码生成")
    prompt = "Once upon a time"
    result = decoder.generate(prompt, max_new_tokens=20)

    print(f"  Prompt: {prompt}")
    print(f"  Generated: {result['generated_text'][:100]}...")
    print(f"  Stats: {result['stats']}")

    # 测试流式生成
    print("\n[3] 测试流式生成")
    print("  Stream output: ", end="")
    for chunk in decoder.stream_generate(prompt, max_new_tokens=10):
        print(chunk, end="", flush=True)
    print()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
