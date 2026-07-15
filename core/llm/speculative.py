"""
AGI Unified Framework - Speculative Decoding
投机解码引擎，支持草稿模型快速生成和目标模型验证
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    FinishReason,
    GenerateParams,
    LLMBackend,
    LLMChunk,
    LLMResponse,
    Message,
    Usage,
)


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
        # 理想情况下每个接受的token不需要目标模型重新生成
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


@dataclass
class DraftToken:
    """草稿Token"""
    token_id: int = 0
    text: str = ""
    log_prob: float = 0.0
    position: int = 0


@dataclass
class VerifyResult:
    """验证结果"""
    accepted_count: int = 0
    rejected_position: int = -1  # -1表示全部接受
    accepted_tokens: List[DraftToken] = field(default_factory=list)
    corrected_token: Optional[DraftToken] = None
    target_log_probs: List[float] = field(default_factory=list)


class MedusaHead:
    """
    Medusa多头预测（模拟）

    Medusa通过训练多个预测头，每个头独立预测未来的多个Token，
    实现并行解码加速。

    这里实现了一个模拟版本，用于演示和测试。
    """

    def __init__(self, num_heads: int = 4, max_predictions: int = 3):
        self._num_heads = num_heads
        self._max_predictions = max_predictions
        self._head_weights = [
            [random.gauss(0, 0.1) for _ in range(100)]
            for _ in range(num_heads)
        ]

    def predict(self, context: str, top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        多头预测

        Args:
            context: 上下文文本
            top_k: 每个头返回的top-k预测

        Returns:
            每个头的预测列表 [(token_text, probability), ...]
        """
        predictions = []
        for head_idx in range(self._num_heads):
            head_preds = []
            # 模拟预测：基于上下文生成伪随机预测
            seed = hash(context + str(head_idx)) % (2**31)
            rng = random.Random(seed)

            for _ in range(self._max_predictions):
                # 生成伪Token
                token_chars = []
                for _ in range(rng.randint(1, 5)):
                    token_chars.append(chr(rng.randint(97, 122)))
                token = "".join(token_chars)

                # 生成伪概率
                prob = rng.random()
                head_preds.append((token, prob))

            # 按概率排序并取top_k
            head_preds.sort(key=lambda x: x[1], reverse=True)
            predictions.append(head_preds[:top_k])

        return predictions

    def combine_predictions(
        self, predictions: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """
        合并多个头的预测

        使用加权投票的方式合并各头的预测结果。

        Args:
            predictions: 各头的预测结果

        Returns:
            合并后的预测列表
        """
        vote_scores: Dict[str, float] = {}
        token_counts: Dict[str, int] = {}

        for head_idx, head_preds in enumerate(predictions):
            weight = 1.0 / (head_idx + 1)  # 越靠后的头权重越低
            for token, prob in head_preds:
                vote_scores[token] = vote_scores.get(token, 0) + prob * weight
                token_counts[token] = token_counts.get(token, 0) + 1

        # 按投票得分排序
        combined = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
        return combined

    def get_num_heads(self) -> int:
        return self._num_heads


class EagleDraftModel:
    """
    Eagle草稿模型（模拟）

    Eagle (Efficient Acceleration of LLM Generation) 使用特征层融合
    来训练高效的草稿模型，在保持高接受率的同时加速推理。

    这里实现了一个模拟版本。
    """

    def __init__(self, hidden_size: int = 512, num_layers: int = 2):
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._feature_weights = [
            [random.gauss(0, 0.1) for _ in range(hidden_size)]
            for _ in range(num_layers)
        ]
        self._vocab_sample = [
            "the", " a", " is", " of", " and", " to", " in", " for",
            " that", " with", " this", " are", " was", " it", " as",
            " be", " at", " have", " from", " or", " an", " by", " not",
            " but", " what", " all", " were", " when", " we", " there",
        ]

    def generate_draft(
        self,
        context: str,
        num_tokens: int = 5,
        temperature: float = 0.8,
    ) -> List[DraftToken]:
        """
        生成草稿Token序列

        Args:
            context: 上下文文本
            num_tokens: 生成的Token数量
            temperature: 温度参数

        Returns:
            草稿Token列表
        """
        tokens = []
        seed = hash(context) % (2**31)
        rng = random.Random(seed)

        for i in range(num_tokens):
            # 模拟特征提取
            features = self._extract_features(context)

            # 模拟Token选择
            if temperature > 0:
                probs = self._softmax(features[:len(self._vocab_sample)], temperature)
            else:
                probs = self._softmax(features[:len(self._vocab_sample)], 0.01)

            # 采样
            idx = rng.choices(range(len(self._vocab_sample)), weights=probs, k=1)[0]
            token_text = self._vocab_sample[idx]
            log_prob = probs[idx] if idx < len(probs) else -10.0

            draft_token = DraftToken(
                token_id=hash(token_text + str(i)) % 100000,
                text=token_text,
                log_prob=log_prob,
                position=i,
            )
            tokens.append(draft_token)
            context += token_text

        return tokens

    def _extract_features(self, text: str) -> List[float]:
        """模拟特征提取"""
        features = []
        vocab_size = len(self._vocab_sample)
        for layer_weights in self._feature_weights:
            layer_feat = 0.0
            for i, char in enumerate(text[-self._hidden_size:]):
                weight_idx = i % len(layer_weights)
                layer_feat += ord(char) * layer_weights[weight_idx]
            features.append(layer_feat)

        # 扩展特征到词汇表大小
        seed = hash(text) % (2**31)
        rng = random.Random(seed)
        while len(features) < vocab_size:
            features.append(rng.gauss(0, 1.0))

        return features[:vocab_size]

    def _softmax(self, values: List[float], temperature: float = 1.0) -> List[float]:
        """计算softmax"""
        if not values:
            return []
        scaled = [v / max(temperature, 0.01) for v in values]
        max_val = max(scaled)
        exp_vals = [pow(2.71828, v - max_val) for v in scaled]
        total = sum(exp_vals)
        return [e / max(total, 1e-10) for e in exp_vals]


class SpeculativeDecoding:
    """
    投机解码引擎

    原理：
    1. 使用草稿模型（小模型）快速生成N个候选Token
    2. 使用目标模型（大模型）并行验证这些Token
    3. 逐Token对比：接受匹配的Token，拒绝第一个不匹配的Token
    4. 从拒绝位置开始，使用目标模型继续生成

    加速效果取决于草稿模型的接受率。接受率越高，加速效果越好。
    """

    def __init__(
        self,
        draft_model: Optional[LLMBackend] = None,
        target_model: Optional[LLMBackend] = None,
        num_speculative_tokens: int = 5,
        use_medusa: bool = False,
        use_eagle: bool = False,
    ):
        self._draft_model = draft_model
        self._target_model = target_model
        self._num_speculative_tokens = num_speculative_tokens
        self._stats = SpeculativeStats()

        # Medusa和Eagle组件
        self._medusa: Optional[MedusaHead] = None
        self._eagle: Optional[EagleDraftModel] = None

        if use_medusa:
            self._medusa = MedusaHead(num_heads=4, max_predictions=num_speculative_tokens)
        if use_eagle:
            self._eagle = EagleDraftModel()

    @property
    def stats(self) -> SpeculativeStats:
        return self._stats

    def draft_generate(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
        num_tokens: int = 0,
    ) -> List[DraftToken]:
        """
        草稿模型快速生成N个候选Token

        Args:
            messages: 消息列表
            params: 生成参数
            num_tokens: 生成的Token数量

        Returns:
            草稿Token列表
        """
        num = num_tokens or self._num_speculative_tokens
        params = params or GenerateParams()
        params.max_tokens = num
        params.temperature = max(params.temperature, 0.5)  # 草稿模型使用较高温度

        start_time = time.time()

        if self._eagle:
            # 使用Eagle草稿模型
            context = " ".join(m.content for m in messages)
            draft_tokens = self._eagle.generate_draft(context, num, params.temperature)
        elif self._medusa:
            # 使用Medusa多头预测
            context = " ".join(m.content for m in messages)
            predictions = self._medusa.predict(context)
            combined = self._medusa.combine_predictions(predictions)
            draft_tokens = []
            for i, (token, prob) in enumerate(combined[:num]):
                draft_tokens.append(DraftToken(
                    token_id=hash(token) % 100000,
                    text=token,
                    log_prob=prob,
                    position=i,
                ))
        elif self._draft_model:
            # 使用实际的草稿模型
            response = self._draft_model.generate(messages, params)
            words = response.content.split()
            draft_tokens = []
            for i, word in enumerate(words[:num]):
                draft_tokens.append(DraftToken(
                    token_id=hash(word) % 100000,
                    text=word,
                    log_prob=-1.0,  # 草稿模型不提供log_prob
                    position=i,
                ))
        else:
            # 无草稿模型，使用Eagle模拟
            self._eagle = EagleDraftModel()
            context = " ".join(m.content for m in messages)
            draft_tokens = self._eagle.generate_draft(context, num, params.temperature)

        self._stats.draft_time += time.time() - start_time
        self._stats.total_draft_tokens += len(draft_tokens)

        return draft_tokens

    def verify_draft(
        self,
        messages: List[Message],
        draft_tokens: List[DraftToken],
        params: Optional[GenerateParams] = None,
    ) -> VerifyResult:
        """
        目标模型并行验证草稿Token

        使用目标模型对草稿Token进行验证，逐Token对比。

        Args:
            messages: 消息列表
            draft_tokens: 草稿Token列表
            params: 生成参数

        Returns:
            VerifyResult: 验证结果
        """
        start_time = time.time()
        params = params or GenerateParams()

        # 构建包含草稿Token的提示
        draft_text = " ".join(t.text for t in draft_tokens)

        # 使用目标模型生成
        if self._target_model:
            verify_params = GenerateParams(
                temperature=0.0,  # 使用低温度以获得确定性输出
                max_tokens=len(draft_tokens) + 1,
            )
            response = self._target_model.generate(messages, verify_params)
            target_text = response.content
        else:
            # 模拟目标模型验证
            target_text = self._simulate_target_verify(messages, draft_tokens)

        # 逐Token对比
        target_words = target_text.split()
        accepted = []
        accepted_count = 0
        rejected_position = -1
        corrected_token = None

        for i, draft_token in enumerate(draft_tokens):
            if i < len(target_words):
                target_word = target_words[i]
                # 简单的匹配：比较文本相似度
                similarity = self._token_similarity(draft_token.text, target_word)

                if similarity >= 0.8:
                    accepted.append(DraftToken(
                        token_id=draft_token.token_id,
                        text=target_word,  # 使用目标模型的输出
                        log_prob=draft_token.log_prob,
                        position=i,
                    ))
                    accepted_count += 1
                else:
                    rejected_position = i
                    corrected_token = DraftToken(
                        token_id=hash(target_word) % 100000,
                        text=target_word,
                        log_prob=-1.0,
                        position=i,
                    )
                    break
            else:
                rejected_position = i
                break

        self._stats.verify_time += time.time() - start_time
        self._stats.total_accepted_tokens += accepted_count
        self._stats.total_rejected_tokens += len(draft_tokens) - accepted_count
        self._stats.total_rounds += 1

        return VerifyResult(
            accepted_count=accepted_count,
            rejected_position=rejected_position,
            accepted_tokens=accepted,
            corrected_token=corrected_token,
        )

    def _token_similarity(self, token1: str, token2: str) -> float:
        """计算两个Token的相似度"""
        if token1 == token2:
            return 1.0
        # 简单的字符级相似度
        common = sum(1 for c in token1 if c in token2)
        total = max(len(token1), len(token2))
        return common / max(total, 1)

    def _simulate_target_verify(
        self, messages: List[Message], draft_tokens: List[DraftToken],
    ) -> str:
        """模拟目标模型验证（当无实际目标模型时）"""
        # 模拟接受率约70%
        result = []
        for token in draft_tokens:
            if random.random() < 0.7:
                result.append(token.text)
            else:
                # 模拟修正
                result.append(token.text + "x")
                break
        return " ".join(result)

    def accept_reject(
        self,
        verify_result: VerifyResult,
        draft_tokens: List[DraftToken],
    ) -> Tuple[str, int]:
        """
        接受/拒绝逻辑

        根据验证结果，构建最终接受的文本和需要重新生成的位置。

        Args:
            verify_result: 验证结果
            draft_tokens: 原始草稿Token

        Returns:
            (accepted_text, resume_position)
        """
        accepted_text = " ".join(t.text for t in verify_result.accepted_tokens)

        if verify_result.rejected_position >= 0:
            # 有拒绝的Token
            resume_position = verify_result.rejected_position
            if verify_result.corrected_token:
                accepted_text += " " + verify_result.corrected_token.text
                resume_position += 1
        else:
            # 全部接受
            resume_position = len(draft_tokens)

        return accepted_text, resume_position

    def generate_with_speculation(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
        max_rounds: int = 10,
    ) -> LLMResponse:
        """
        投机解码生成

        完整的投机解码流程：
        1. 草稿模型生成候选Token
        2. 目标模型验证
        3. 接受/拒绝
        4. 重复直到完成

        Args:
            messages: 消息列表
            params: 生成参数
            max_rounds: 最大轮数

        Returns:
            LLMResponse: 最终响应
        """
        start_time = time.time()
        params = params or GenerateParams()
        all_accepted_text = ""
        total_target_calls = 0

        for round_num in range(max_rounds):
            # 1. 草稿生成
            draft_tokens = self.draft_generate(messages, params)
            if not draft_tokens:
                break

            # 2. 验证
            verify_result = self.verify_draft(messages, draft_tokens, params)
            total_target_calls += 1

            # 3. 接受/拒绝
            accepted_text, resume_pos = self.accept_reject(verify_result, draft_tokens)
            all_accepted_text += (" " if all_accepted_text else "") + accepted_text

            # 4. 检查是否完成
            if verify_result.rejected_position < 0 and len(draft_tokens) < self._num_speculative_tokens:
                # 草稿模型生成了少于预期的Token，可能已到达结束
                break

            if resume_pos >= len(draft_tokens):
                # 全部接受，继续下一轮
                # 更新消息上下文
                assistant_msg = Message(role="assistant", content=all_accepted_text)
                messages = list(messages) + [assistant_msg]
                continue
            else:
                # 有拒绝，从拒绝位置继续
                break

        # 最终使用目标模型生成剩余部分
        if self._target_model and resume_pos < len(draft_tokens):
            final_params = GenerateParams(
                temperature=params.temperature,
                max_tokens=params.max_tokens,
            )
            final_response = self._target_model.generate(messages, final_params)
            total_target_calls += 1

            self._stats.total_target_tokens += final_response.usage.completion_tokens

            full_content = all_accepted_text
            if final_response.content:
                full_content += (" " if full_content else "") + final_response.content

            self._stats.total_time = time.time() - start_time

            return LLMResponse(
                content=full_content,
                usage=Usage(
                    prompt_tokens=final_response.usage.prompt_tokens,
                    completion_tokens=len(full_content.split()),
                    total_tokens=final_response.usage.prompt_tokens + len(full_content.split()),
                ),
                finish_reason=final_response.finish_reason,
                model=final_response.model,
                metadata={
                    "speculative_decoding": True,
                    "speculative_stats": self._stats.to_dict(),
                    "target_model_calls": total_target_calls,
                },
            )

        self._stats.total_time = time.time() - start_time
        self._stats.total_target_tokens = total_target_calls

        return LLMResponse(
            content=all_accepted_text,
            usage=Usage(
                completion_tokens=len(all_accepted_text.split()),
                total_tokens=len(all_accepted_text.split()),
            ),
            finish_reason=FinishReason.STOP,
            metadata={
                "speculative_decoding": True,
                "speculative_stats": self._stats.to_dict(),
                "target_model_calls": total_target_calls,
            },
        )

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = SpeculativeStats()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.to_dict()
