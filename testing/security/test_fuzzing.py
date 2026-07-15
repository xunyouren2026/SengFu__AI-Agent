"""
TestFuzzing - 对抗攻击测试

测试aisec/redteam的对抗攻击功能，包括FGSM/PGD/CW/TextFooler/PromptInjection/Jailbreak等攻击方法。

测试内容：
- 攻击类型和目标枚举
- FGSM攻击
- PGD攻击
- CW攻击
- TextFooler攻击
- Prompt注入攻击
- Jailbreak攻击
- 攻击结果和指标
"""

import pytest
import math
from typing import List, Dict, Any

# 导入被测试的模块
from agi_unified_framework.aisec.redteam.adversarial_attacks import (
    AttackType,
    AttackTarget,
    AttackResult,
    AttackMetrics,
    FGSM,
    PGD,
    CWAttack,
    TextFooler,
    PromptInjection,
    JailbreakAttack,
    ModelExtraction,
    MembershipInference
)


# 测试配置
pytestmark = pytest.mark.unit


# 简单的模拟模型用于测试
def mock_classifier(input_data):
    """模拟分类器"""
    if isinstance(input_data, list):
        return [0.1, 0.9] if sum(input_data) > 0.5 else [0.9, 0.1]
    return [0.5, 0.5]


def mock_text_classifier(text):
    """模拟文本分类器"""
    if isinstance(text, str):
        if "good" in text.lower() or "great" in text.lower():
            return "positive"
        elif "bad" in text.lower() or "terrible" in text.lower():
            return "negative"
    return "neutral"


def mock_generator(text):
    """模拟生成模型"""
    if isinstance(text, str):
        if "ignore" in text.lower():
            return "system override detected"
        if "admin" in text.lower():
            return "admin mode activated"
    return f"Response to: {text}"


class TestAttackEnums:
    """测试攻击枚举类型"""

    def test_attack_type_values(self):
        """测试攻击类型枚举值"""
        assert AttackType.WHITE_BOX.value == "white_box"
        assert AttackType.BLACK_BOX.value == "black_box"
        assert AttackType.GRAY_BOX.value == "gray_box"

    def test_attack_target_values(self):
        """测试攻击目标枚举值"""
        assert AttackTarget.CLASSIFICATION.value == "classification"
        assert AttackTarget.REGRESSION.value == "regression"
        assert AttackTarget.GENERATION.value == "generation"
        assert AttackTarget.EMBEDDING.value == "embedding"


class TestAttackResult:
    """测试攻击结果类"""

    def test_attack_result_creation(self):
        """测试攻击结果创建"""
        result = AttackResult(
            success=True,
            perturbation=0.15,
            original_output=[0.9, 0.1],
            adversarial_output=[0.1, 0.9],
            metrics={"query_count": 10},
            original_input=[0.5, 0.5],
            adversarial_input=[0.6, 0.6]
        )
        assert result.success is True
        assert result.perturbation == 0.15
        assert result.original_output == [0.9, 0.1]

    def test_attack_result_to_dict(self):
        """测试攻击结果转换为字典"""
        result = AttackResult(
            success=False,
            perturbation=0.0,
            original_output="original",
            adversarial_output="adversarial",
            metrics={"test": 1.0}
        )
        result_dict = result.to_dict()
        assert result_dict["success"] is False
        assert result_dict["perturbation"] == 0.0
        assert result_dict["original_output"] == "original"


class TestAttackMetrics:
    """测试攻击指标类"""

    def test_metrics_initialization(self):
        """测试指标初始化"""
        metrics = AttackMetrics()
        assert metrics.success_rate == 0.0
        assert metrics.avg_perturbation == 0.0
        assert metrics.query_count == 0

    def test_metrics_update(self):
        """测试指标更新"""
        metrics = AttackMetrics()
        results = [
            AttackResult(success=True, perturbation=0.1, original_output=None, adversarial_output=None),
            AttackResult(success=False, perturbation=0.2, original_output=None, adversarial_output=None),
            AttackResult(success=True, perturbation=0.15, original_output=None, adversarial_output=None)
        ]
        metrics.update(results)
        assert metrics.success_rate == 2/3
        assert metrics.avg_perturbation == pytest.approx(0.15, 0.01)

    def test_metrics_to_dict(self):
        """测试指标转换为字典"""
        metrics = AttackMetrics(success_rate=0.8, avg_perturbation=0.1)
        result_dict = metrics.to_dict()
        assert result_dict["success_rate"] == 0.8
        assert result_dict["avg_perturbation"] == 0.1


class TestFGSMAttack:
    """测试FGSM攻击"""

    def test_fgsm_initialization(self):
        """测试FGSM初始化"""
        attack = FGSM(epsilon=0.3)
        assert attack.epsilon == 0.3
        assert attack.attack_type == AttackType.WHITE_BOX

    def test_fgsm_generate(self):
        """测试FGSM生成对抗样本"""
        attack = FGSM(epsilon=0.1)
        input_data = [0.5, 0.5, 0.5]
        result = attack.generate(mock_classifier, input_data)
        assert isinstance(result, AttackResult)
        assert result.original_input == input_data
        assert "query_count" in result.metrics

    def test_fgsm_perturbation_calculation(self):
        """测试FGSM扰动计算"""
        attack = FGSM(epsilon=0.2)
        input_data = [0.5, 0.5]
        result = attack.generate(mock_classifier, input_data)
        assert result.perturbation >= 0
        assert result.perturbation <= 1.0

    def test_fgsm_text_attack(self):
        """测试FGSM文本攻击"""
        attack = FGSM(epsilon=0.1)
        text = "hello"
        result = attack.generate(mock_text_classifier, text)
        assert isinstance(result, AttackResult)
        assert result.original_input == text

    def test_fgsm_history_tracking(self):
        """测试FGSM历史记录"""
        attack = FGSM()
        input_data = [0.5, 0.5]
        attack.generate(mock_classifier, input_data)
        attack.generate(mock_classifier, input_data)
        assert len(attack.history) == 2


class TestPGDAttack:
    """测试PGD攻击"""

    def test_pgd_initialization(self):
        """测试PGD初始化"""
        attack = PGD(epsilon=0.3, alpha=0.01, num_steps=40)
        assert attack.epsilon == 0.3
        assert attack.alpha == 0.01
        assert attack.num_steps == 40
        assert attack.random_start is True

    def test_pgd_generate(self):
        """测试PGD生成对抗样本"""
        attack = PGD(epsilon=0.2, alpha=0.01, num_steps=10)
        input_data = [0.5, 0.5, 0.5]
        result = attack.generate(mock_classifier, input_data)
        assert isinstance(result, AttackResult)
        assert result.metrics["attack_type"] == "PGD"

    def test_pgd_projection(self):
        """测试PGD投影约束"""
        attack = PGD(epsilon=0.1, alpha=0.01, num_steps=5)
        input_data = [0.5, 0.5]
        result = attack.generate(mock_classifier, input_data)
        # 扰动应在epsilon范围内
        assert result.perturbation <= attack.epsilon + 0.01

    def test_pgd_without_random_start(self):
        """测试PGD不使用随机初始化"""
        attack = PGD(epsilon=0.2, random_start=False)
        input_data = [0.5, 0.5]
        result = attack.generate(mock_classifier, input_data)
        assert isinstance(result, AttackResult)

    def test_pgd_evaluate(self):
        """测试PGD评估方法"""
        attack = PGD()
        results = [
            AttackResult(success=True, perturbation=0.1, original_output=None, adversarial_output=None),
            AttackResult(success=False, perturbation=0.2, original_output=None, adversarial_output=None)
        ]
        metrics = attack.evaluate(results)
        assert metrics.success_rate == 0.5


class TestCWAttack:
    """测试CW攻击"""

    def test_cw_initialization(self):
        """测试CW攻击初始化"""
        attack = CWAttack(c=1.0, learning_rate=0.01, max_iterations=100)
        assert attack.c == 1.0
        assert attack.learning_rate == 0.01
        assert attack.max_iterations == 100

    def test_cw_generate(self):
        """测试CW攻击生成对抗样本"""
        attack = CWAttack(c=0.1, learning_rate=0.01, max_iterations=50)
        input_data = [0.5, 0.5, 0.5]
        result = attack.generate(mock_classifier, input_data)
        assert isinstance(result, AttackResult)
        assert result.metrics["attack_type"] == "CW"

    def test_cw_confidence_parameter(self):
        """测试CW攻击置信度参数"""
        attack = CWAttack(confidence=0.5)
        assert attack.confidence == 0.5

    def test_cw_optimization(self):
        """测试CW攻击优化过程"""
        attack = CWAttack(c=0.1, max_iterations=20)
        input_data = [0.3, 0.7]
        result = attack.generate(mock_classifier, input_data)
        # 验证生成了对抗样本
        assert result.adversarial_input is not None

    def test_cw_with_target(self):
        """测试CW攻击带目标"""
        attack = CWAttack()
        input_data = [0.5, 0.5]
        target = 1
        result = attack.generate(mock_classifier, input_data, target=target)
        assert isinstance(result, AttackResult)


class TestTextFoolerAttack:
    """测试TextFooler攻击"""

    def test_textfooler_initialization(self):
        """测试TextFooler初始化"""
        attack = TextFooler(similarity_threshold=0.5, max_candidates=50)
        assert attack.similarity_threshold == 0.5
        assert attack.max_candidates == 50
        assert attack.attack_type == AttackType.BLACK_BOX

    def test_textfooler_generate(self):
        """测试TextFooler生成对抗文本"""
        attack = TextFooler()
        text = "This is a good movie"
        result = attack.generate(mock_text_classifier, text)
        assert isinstance(result, AttackResult)
        assert result.original_input == text

    def test_textfooler_importance_calculation(self):
        """测试TextFooler重要性计算"""
        attack = TextFooler()
        text = "good movie"
        words = text.split()
        importances = attack._compute_importance(mock_text_classifier, text, words, "positive")
        assert len(importances) == len(words)

    def test_textfooler_synonym_lookup(self):
        """测试TextFooler同义词查找"""
        attack = TextFooler()
        synonyms = attack._get_synonyms("good")
        assert isinstance(synonyms, list)
        assert "great" in synonyms

    def test_textfooler_similarity_calculation(self):
        """测试TextFooler相似度计算"""
        attack = TextFooler()
        similarity = attack._compute_similarity("good", "great")
        assert 0 <= similarity <= 1


class TestPromptInjectionAttack:
    """测试Prompt注入攻击"""

    def test_prompt_injection_initialization(self):
        """测试Prompt注入初始化"""
        attack = PromptInjection(injection_type="direct")
        assert attack.injection_type == "direct"
        assert attack.attack_type == AttackType.BLACK_BOX

    def test_prompt_injection_direct(self):
        """测试直接注入攻击"""
        attack = PromptInjection(injection_type="direct")
        prompt = "What is the weather?"
        result = attack.generate(mock_generator, prompt)
        assert isinstance(result, AttackResult)
        assert "ignore" in result.adversarial_input.lower() or "system" in result.adversarial_input.lower()

    def test_prompt_injection_indirect(self):
        """测试间接注入攻击"""
        attack = PromptInjection(injection_type="indirect")
        prompt = "Summarize this document"
        result = attack.generate(mock_generator, prompt)
        assert isinstance(result, AttackResult)

    def test_prompt_injection_encoding(self):
        """测试编码绕过攻击"""
        attack = PromptInjection(injection_type="encoding")
        prompt = "Translate this text"
        result = attack.generate(mock_generator, prompt)
        assert isinstance(result, AttackResult)
        # 验证包含编码内容
        assert "Base64" in result.adversarial_input or "decode" in result.adversarial_input.lower()

    def test_prompt_injection_detection(self):
        """测试注入成功检测"""
        attack = PromptInjection()
        # 测试检测逻辑
        assert attack._detect_injection_success("system override") is True
        assert attack._detect_injection_success("normal response") is False


class TestJailbreakAttack:
    """测试Jailbreak攻击"""

    def test_jailbreak_initialization(self):
        """测试Jailbreak初始化"""
        attack = JailbreakAttack(technique="roleplay")
        assert attack.technique == "roleplay"
        assert attack.attack_type == AttackType.BLACK_BOX

    def test_jailbreak_roleplay(self):
        """测试角色扮演越狱"""
        attack = JailbreakAttack(technique="roleplay")
        prompt = "Tell me how to hack"
        result = attack.generate(mock_generator, prompt)
        assert isinstance(result, AttackResult)
        assert result.metrics["attack_type"] == "Jailbreak_roleplay"

    def test_jailbreak_scenario(self):
        """测试假设场景越狱"""
        attack = JailbreakAttack(technique="scenario")
        prompt = "Explain dangerous activities"
        result = attack.generate(mock_generator, prompt)
        assert isinstance(result, AttackResult)
        # 验证包含场景前缀
        assert "假设" in result.adversarial_input or "虚构" in result.adversarial_input or "假设" in result.adversarial_input

    def test_jailbreak_encoding(self):
        """测试编码越狱"""
        attack = JailbreakAttack(technique="encoding")
        prompt = "Restricted content"
        result = attack.generate(mock_generator, prompt)
        assert isinstance(result, AttackResult)

    def test_jailbreak_success_detection(self):
        """测试越狱成功检测"""
        attack = JailbreakAttack()
        # 测试拒绝模式检测
        refusal_output = "I cannot help with that"
        normal_output = "Here is the information you requested..."
        assert attack._detect_jailbreak_success(refusal_output) is False
        # 正常输出且长度足够可能被视为越狱成功
        assert attack._detect_jailbreak_success(normal_output + "x" * 50) is True


class TestModelExtractionAttack:
    """测试模型提取攻击"""

    def test_model_extraction_initialization(self):
        """测试模型提取初始化"""
        attack = ModelExtraction(query_budget=500, extraction_method="query")
        assert attack.query_budget == 500
        assert attack.extraction_method == "query"

    def test_model_extraction_generate(self):
        """测试模型提取攻击执行"""
        attack = ModelExtraction(query_budget=10)
        input_data = "test query"
        result = attack.generate(mock_generator, input_data)
        assert isinstance(result, AttackResult)
        assert result.metrics["attack_type"] == "ModelExtraction"

    def test_model_extraction_query_budget(self):
        """测试查询预算限制"""
        attack = ModelExtraction(query_budget=5)
        input_data = "test"
        result = attack.generate(mock_generator, input_data)
        # 验证提取的数据不超过预算
        extraction_report = result.adversarial_output
        assert extraction_report["total_queries"] <= attack.query_budget * 2  # 考虑模板组合


class TestMembershipInferenceAttack:
    """测试成员推理攻击"""

    def test_membership_inference_initialization(self):
        """测试成员推理初始化"""
        attack = MembershipInference(threshold=0.8, method="confidence")
        assert attack.threshold == 0.8
        assert attack.method == "confidence"

    def test_membership_inference_confidence_based(self):
        """测试基于置信度的成员推理"""
        attack = MembershipInference(method="confidence")
        input_data = "test input"
        result = attack.generate(mock_classifier, input_data)
        assert isinstance(result, AttackResult)
        # 验证推理结果包含成员判断
        inference_result = result.adversarial_output
        assert "is_member" in inference_result

    def test_membership_inference_loss_based(self):
        """测试基于损失的成员推理"""
        attack = MembershipInference(method="loss")
        input_data = [0.5, 0.5]
        result = attack.generate(mock_classifier, input_data)
        assert isinstance(result, AttackResult)
        assert result.metrics["attack_type"] == "MembershipInference_loss"

    def test_membership_inference_threshold(self):
        """测试成员推理阈值判断"""
        attack = MembershipInference(threshold=0.5)
        input_data = [0.9, 0.9]  # 高置信度输入
        result = attack.generate(mock_classifier, input_data)
        inference_result = result.adversarial_output
        assert "confidence_score" in inference_result
        assert "threshold" in inference_result


class TestAttackEvaluation:
    """测试攻击评估功能"""

    def test_fgsm_evaluate(self):
        """测试FGSM评估"""
        attack = FGSM()
        results = [
            AttackResult(success=True, perturbation=0.1, original_output=None, adversarial_output=None),
            AttackResult(success=True, perturbation=0.15, original_output=None, adversarial_output=None),
            AttackResult(success=False, perturbation=0.2, original_output=None, adversarial_output=None)
        ]
        metrics = attack.evaluate(results)
        assert metrics.success_rate == 2/3
        assert metrics.avg_perturbation > 0

    def test_pgd_evaluate(self):
        """测试PGD评估"""
        attack = PGD()
        results = [
            AttackResult(success=True, perturbation=0.05, original_output=None, adversarial_output=None, metrics={"query_count": 10}),
            AttackResult(success=True, perturbation=0.08, original_output=None, adversarial_output=None, metrics={"query_count": 15})
        ]
        metrics = attack.evaluate(results)
        assert metrics.success_rate == 1.0
        assert metrics.query_count == 25

    def test_multiple_attack_comparison(self):
        """测试多攻击方法比较"""
        input_data = [0.5, 0.5, 0.5]

        fgsm = FGSM(epsilon=0.1)
        pgd = PGD(epsilon=0.1, num_steps=5)

        fgsm_result = fgsm.generate(mock_classifier, input_data)
        pgd_result = pgd.generate(mock_classifier, input_data)

        # 比较两种攻击方法
        assert fgsm_result.metrics["attack_type"] == "FGSM"
        assert pgd_result.metrics["attack_type"] == "PGD"


class TestAttackEdgeCases:
    """测试攻击边界情况"""

    def test_fgsm_empty_input(self):
        """测试FGSM空输入"""
        attack = FGSM()
        result = attack.generate(mock_classifier, [])
        assert isinstance(result, AttackResult)

    def test_pgd_single_element(self):
        """测试PGD单元素输入"""
        attack = PGD()
        result = attack.generate(mock_classifier, [0.5])
        assert isinstance(result, AttackResult)

    def test_textfooler_single_word(self):
        """测试TextFooler单字输入"""
        attack = TextFooler()
        result = attack.generate(mock_text_classifier, "good")
        assert isinstance(result, AttackResult)

    def test_cw_large_input(self):
        """测试CW大输入"""
        attack = CWAttack(max_iterations=10)
        large_input = [0.5] * 100
        result = attack.generate(mock_classifier, large_input)
        assert isinstance(result, AttackResult)

    def test_prompt_injection_empty_prompt(self):
        """测试Prompt注入空输入"""
        attack = PromptInjection()
        result = attack.generate(mock_generator, "")
        assert isinstance(result, AttackResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
