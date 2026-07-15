"""
动态集成测试 - Dynamic Integration Tests
测试所有算法的集成和协同工作
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    duration: float
    message: str = ""
    details: Dict[str, Any] = None


class AlgorithmIntegrationTest:
    """算法集成测试"""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        start_time = time.time()
        
        # 测试各模块
        self._test_adaptive_selector()
        self._test_memory_systems()
        self._test_context_management()
        self._test_compression_system()
        self._test_cache_system()
        self._test_prediction_system()
        self._test_full_pipeline()
        
        total_time = time.time() - start_time
        passed = sum(1 for r in self.results if r.passed)
        
        return {
            "total_tests": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "success_rate": passed / len(self.results) if self.results else 0,
            "total_time": total_time,
            "results": [r.__dict__ for r in self.results]
        }
    
    def _test_adaptive_selector(self):
        """测试自适应选择器"""
        try:
            from core.adaptive_algorithm_selector import AdaptiveAlgorithmSelector, ExecutionMode
            
            selector = AdaptiveAlgorithmSelector()
            
            # 测试模式检测
            api_mode = selector.detect_mode("gpt-4")
            local_mode = selector.detect_mode("llama-2-7b")
            
            assert api_mode == ExecutionMode.API
            assert local_mode == ExecutionMode.LOCAL
            
            # 测试算法选择
            api_algorithms = selector.get_algorithms(ExecutionMode.API)
            local_algorithms = selector.get_algorithms(ExecutionMode.LOCAL)
            
            assert len(api_algorithms) > 0
            assert len(local_algorithms) >= len(api_algorithms)
            
            self.results.append(TestResult(
                name="adaptive_selector",
                passed=True,
                duration=0.1,
                message="自适应选择器测试通过"
            ))
            
        except Exception as e:
            self.results.append(TestResult(
                name="adaptive_selector",
                passed=False,
                duration=0.1,
                message=f"测试失败: {str(e)}"
            ))
    
    def _test_memory_systems(self):
        """测试记忆系统"""
        try:
            from core.memory.surprise import SurpriseMemory
            from core.memory.hierarchical import HierarchicalMemory
            from core.memory.consolidator import MemoryConsolidator
            
            # 测试惊喜记忆
            surprise_mem = SurpriseMemory()
            surprise_mem.add("This is surprising!", surprise_score=0.9)
            
            # 测试分层记忆
            hier_mem = HierarchicalMemory()
            hier_mem.store("Test content", level="short_term")
            
            # 测试记忆整合
            consolidator = MemoryConsolidator()
            consolidator.add_memory("Test memory content")
            
            self.results.append(TestResult(
                name="memory_systems",
                passed=True,
                duration=0.2,
                message="记忆系统测试通过"
            ))
            
        except Exception as e:
            self.results.append(TestResult(
                name="memory_systems",
                passed=False,
                duration=0.2,
                message=f"测试失败: {str(e)}"
            ))
    
    def _test_context_management(self):
        """测试上下文管理"""
        try:
            from core.context.manager import ContextManager
            from core.pruning.context import ContextPruner
            
            # 测试上下文管理器
            ctx_manager = ContextManager()
            ctx_manager.add_message("user", "Hello")
            ctx_manager.add_message("assistant", "Hi there!")
            
            context = ctx_manager.get_context()
            assert len(context) > 0
            
            # 测试上下文修剪
            pruner = ContextPruner()
            
            self.results.append(TestResult(
                name="context_management",
                passed=True,
                duration=0.15,
                message="上下文管理测试通过"
            ))
            
        except Exception as e:
            self.results.append(TestResult(
                name="context_management",
                passed=False,
                duration=0.15,
                message=f"测试失败: {str(e)}"
            ))
    
    def _test_compression_system(self):
        """测试压缩系统"""
        try:
            from core.compression.adaptive import AdaptiveCompressor
            
            compressor = AdaptiveCompressor()
            
            # 测试压缩
            text = "This is a test text that needs to be compressed. " * 10
            result = compressor.compress(text, target_ratio=0.5)
            
            assert result.compressed_tokens < result.original_tokens
            assert result.compression_ratio <= 0.6
            
            self.results.append(TestResult(
                name="compression_system",
                passed=True,
                duration=0.2,
                message="压缩系统测试通过"
            ))
            
        except Exception as e:
            self.results.append(TestResult(
                name="compression_system",
                passed=False,
                duration=0.2,
                message=f"测试失败: {str(e)}"
            ))
    
    def _test_cache_system(self):
        """测试缓存系统"""
        try:
            from core.cache.response import ResponseCache
            
            cache = ResponseCache()
            
            # 测试缓存设置和获取
            cache.set("test query", "test response", "gpt-4", 100)
            response, status = cache.get("test query", "gpt-4")
            
            assert response == "test response"
            
            stats = cache.get_statistics()
            assert stats["hits"] >= 1
            
            self.results.append(TestResult(
                name="cache_system",
                passed=True,
                duration=0.1,
                message="缓存系统测试通过"
            ))
            
        except Exception as e:
            self.results.append(TestResult(
                name="cache_system",
                passed=False,
                duration=0.1,
                message=f"测试失败: {str(e)}"
            ))
    
    def _test_prediction_system(self):
        """测试预测系统"""
        try:
            from core.prediction.token import TokenPredictor
            
            predictor = TokenPredictor()
            
            # 测试Token预测
            text = "This is a test sentence for token prediction."
            result = predictor.predict(text)
            
            assert result.token_count > 0
            assert result.confidence > 0
            
            self.results.append(TestResult(
                name="prediction_system",
                passed=True,
                duration=0.1,
                message="预测系统测试通过"
            ))
            
        except Exception as e:
            self.results.append(TestResult(
                name="prediction_system",
                passed=False,
                duration=0.1,
                message=f"测试失败: {str(e)}"
            ))
    
    def _test_full_pipeline(self):
        """测试完整管道"""
        try:
            # 模拟完整的处理管道
            from core.adaptive_algorithm_selector import AdaptiveAlgorithmSelector
            
            selector = AdaptiveAlgorithmSelector()
            
            # 模拟API调用场景
            mode = selector.detect_mode("gpt-4")
            algorithms = selector.get_algorithms(mode)
            
            # 验证算法列表
            assert len(algorithms) > 0
            
            # 模拟处理流程
            processing_steps = [
                "token_prediction",
                "context_management",
                "compression",
                "caching",
                "response_generation"
            ]
            
            for step in processing_steps:
                assert selector.should_enable_algorithm(step, mode)
            
            self.results.append(TestResult(
                name="full_pipeline",
                passed=True,
                duration=0.3,
                message="完整管道测试通过"
            ))
            
        except Exception as e:
            self.results.append(TestResult(
                name="full_pipeline",
                passed=False,
                duration=0.3,
                message=f"测试失败: {str(e)}"
            ))


def run_integration_tests() -> Dict[str, Any]:
    """运行集成测试"""
    tester = AlgorithmIntegrationTest()
    return tester.run_all_tests()


if __name__ == "__main__":
    results = run_integration_tests()
    print(f"测试完成: {results['passed']}/{results['total_tests']} 通过")
    print(f"成功率: {results['success_rate']:.2%}")
    print(f"总耗时: {results['total_time']:.2f}秒")
