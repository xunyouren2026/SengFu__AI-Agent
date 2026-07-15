"""
UFO AGI 框架 - 自适应算法选择器
根据运行模式自动选择算法组合

支持两种模式:
1. API模式 (10个算法) - 调用外部模型时使用
2. LOCAL模式 (25个算法) - 本地模型时使用

作者: UFO Framework Team
日期: 2025-05-19
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """执行模式枚举"""
    AUTO = auto()      # 自动检测
    API = auto()       # API调用模式 (10算法)
    LOCAL = auto()     # 本地模型模式 (25算法)


@dataclass
class AlgorithmConfig:
    """算法配置"""
    name: str
    module_path: str
    class_name: str
    enabled_in_api: bool
    enabled_in_local: bool
    priority: int  # 1-10, 数字越小优先级越高
    dependencies: List[str]


class AdaptiveAlgorithmSelector:
    """
    自适应算法选择器
    
    根据当前执行环境自动选择最优算法组合
    """
    
    # 算法注册表 - 共35个算法
    ALGORITHMS: Dict[str, AlgorithmConfig] = {
        # ========== API模式算法 (10个) ==========
        "comi_compressor": AlgorithmConfig(
            name="COMI框架",
            module_path="core.context.comi",
            class_name="COMICompressor",
            enabled_in_api=True,
            enabled_in_local=True,
            priority=1,
            dependencies=[]
        ),
        "context_manager": AlgorithmConfig(
            name="上下文管理器",
            module_path="core.context.manager",
            class_name="ContextManager",
            enabled_in_api=True,
            enabled_in_local=True,
            priority=1,
            dependencies=[]
        ),
        "surprise_memory": AlgorithmConfig(
            name="惊喜度记忆",
            module_path="core.memory.surprise",
            class_name="SurpriseBasedMemory",
            enabled_in_api=True,
            enabled_in_local=True,
            priority=1,
            dependencies=["context_manager"]
        ),
        "hierarchical_memory": AlgorithmConfig(
            name="分层记忆",
            module_path="core.memory.hierarchical",
            class_name="HierarchicalMemory",
            enabled_in_api=True,
            enabled_in_local=True,
            priority=2,
            dependencies=[]
        ),
        "quality_scorer": AlgorithmConfig(
            name="质量评分",
            module_path="core.quality.scorer",
            class_name="QualityScorer",
            enabled_in_api=True,
            enabled_in_local=True,
            priority=2,
            dependencies=[]
        ),
        "rag_retriever": AlgorithmConfig(
            name="RAG检索",
            module_path="core.rag.retriever",
            class_name="RAGRetriever",
            enabled_in_api=True,
            enabled_in_local=True,
            priority=2,
            dependencies=[]
        ),
        "dialog_summarizer": AlgorithmConfig(
            name="对话摘要",
            module_path="core.summarization.dialog",
            class_name="DialogSummarizer",
            enabled_in_api=True,
            enabled_in_local=True,
            priority=3,
            dependencies=["context_manager"]
        ),
        "multi_turn_optimizer": AlgorithmConfig(
            name="多轮优化",
            module_path="core.optimization.multi_turn",
            class_name="MultiTurnOptimizer",
            enabled_in_api=True,
            enabled_in_local=True,
            priority=3,
            dependencies=[]
        ),
        "prompt_templates": AlgorithmConfig(
            name="提示模板",
            module_path="core.prompt.templates",
            class_name="PromptTemplateManager",
            enabled_in_api=True,
            enabled_in_local=True,
            priority=3,
            dependencies=[]
        ),
        "cost_monitor": AlgorithmConfig(
            name="成本监控",
            module_path="core.monitoring.cost",
            class_name="CostMonitor",
            enabled_in_api=True,
            enabled_in_local=False,  # 本地模式不需要
            priority=4,
            dependencies=[]
        ),
        
        # ========== 本地模式额外算法 (15个) ==========
        "infini_attention": AlgorithmConfig(
            name="Infini-Attention",
            module_path="core.attention.infini",
            class_name="InfiniAttention",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=1,
            dependencies=[]
        ),
        "gist_attention": AlgorithmConfig(
            name="Gist Sparse Attention",
            module_path="core.attention.gist",
            class_name="GistSparseAttention",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=1,
            dependencies=[]
        ),
        "speculative_decode": AlgorithmConfig(
            name="投机解码",
            module_path="core.inference.speculative",
            class_name="SpeculativeDecoder",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=1,
            dependencies=[]
        ),
        "snapkv": AlgorithmConfig(
            name="SnapKV",
            module_path="core.kv_cache.snapkv",
            class_name="SnapKVCache",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=1,
            dependencies=[]
        ),
        "ttt_layer": AlgorithmConfig(
            name="TTT层",
            module_path="core.layers.ttt",
            class_name="TTTLayer",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=2,
            dependencies=[]
        ),
        "memgpt_memory": AlgorithmConfig(
            name="MemGPT分级记忆",
            module_path="core.memory.memgpt",
            class_name="MemGPTMemory",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=2,
            dependencies=["hierarchical_memory"]
        ),
        "motion_adaptive": AlgorithmConfig(
            name="运动自适应权重",
            module_path="core.video.motion",
            class_name="MotionAdaptiveWeight",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=2,
            dependencies=[]
        ),
        "adaptive_tile": AlgorithmConfig(
            name="自适应Tile融合",
            module_path="core.video.tile",
            class_name="AdaptiveTileFusion",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=2,
            dependencies=[]
        ),
        "delta_mem": AlgorithmConfig(
            name="δ-mem在线校正",
            module_path="core.memory.delta",
            class_name="DeltaMemory",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=2,
            dependencies=[]
        ),
        "mini_cache": AlgorithmConfig(
            name="MiniCache",
            module_path="core.kv_cache.mini",
            class_name="MiniCache",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=2,
            dependencies=[]
        ),
        "modular_connector": AlgorithmConfig(
            name="Modular Connector",
            module_path="core.multimodal.connector",
            class_name="ModularConnector",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=3,
            dependencies=[]
        ),
        "consistency_loss": AlgorithmConfig(
            name="记忆一致性损失",
            module_path="core.training.consistency",
            class_name="ConsistencyLoss",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=3,
            dependencies=[]
        ),
        "k_token_merge": AlgorithmConfig(
            name="K-Token Merging",
            module_path="core.compression.k_token",
            class_name="KTokenMerger",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=3,
            dependencies=[]
        ),
        "gist_tokens": AlgorithmConfig(
            name="Gist Tokens",
            module_path="core.compression.gist_tokens",
            class_name="GistTokens",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=3,
            dependencies=[]
        ),
        "polar_quant": AlgorithmConfig(
            name="PolarQuant",
            module_path="core.quantization.polar",
            class_name="PolarQuantizer",
            enabled_in_api=False,
            enabled_in_local=True,
            priority=4,
            dependencies=[]
        ),
    }
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.AUTO):
        """
        初始化选择器
        
        Args:
            mode: 执行模式，AUTO会自动检测
        """
        self.mode = mode
        self._detected_mode: Optional[ExecutionMode] = None
        self._active_algorithms: Dict[str, Any] = {}
        self._algorithm_instances: Dict[str, Any] = {}
        
        if mode == ExecutionMode.AUTO:
            self._detected_mode = self._detect_execution_mode()
            logger.info(f"自动检测到执行模式: {self._detected_mode.name}")
        else:
            self._detected_mode = mode
            logger.info(f"手动设置执行模式: {mode.name}")
    
    def _detect_execution_mode(self) -> ExecutionMode:
        """
        自动检测执行模式
        
        检测逻辑:
        1. 检查是否有本地模型配置文件
        2. 检查环境变量 MODEL_MODE
        3. 检查是否有GPU且安装了transformers
        4. 默认使用API模式（更安全）
        """
        import os
        
        # 检查环境变量
        env_mode = os.getenv("UFO_MODEL_MODE", "").upper()
        if env_mode == "LOCAL":
            return ExecutionMode.LOCAL
        elif env_mode == "API":
            return ExecutionMode.API
        
        # 检查是否有本地模型
        try:
            import torch
            import transformers
            
            # 检查是否有本地模型权重
            local_model_path = os.getenv("UFO_LOCAL_MODEL_PATH", "")
            if local_model_path and os.path.exists(local_model_path):
                logger.info(f"检测到本地模型: {local_model_path}")
                return ExecutionMode.LOCAL
            
            # 检查GPU可用性
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory > 20:  # 大于20GB显存，可能是本地模型
                    logger.info(f"检测到高显存GPU: {gpu_memory:.1f}GB，启用本地模式")
                    return ExecutionMode.LOCAL
        except ImportError:
            pass
        
        # 默认使用API模式
        logger.info("未检测到本地模型环境，使用API模式")
        return ExecutionMode.API
    
    def get_active_algorithms(self) -> List[str]:
        """
        获取当前模式下启用的算法列表
        
        Returns:
            算法名称列表（按优先级排序）
        """
        if self._detected_mode == ExecutionMode.LOCAL:
            # 本地模式：启用全部25个算法
            enabled = [
                name for name, config in self.ALGORITHMS.items()
                if config.enabled_in_local
            ]
        else:
            # API模式：启用10个算法
            enabled = [
                name for name, config in self.ALGORITHMS.items()
                if config.enabled_in_api
            ]
        
        # 按优先级排序
        enabled.sort(
            key=lambda x: self.ALGORITHMS[x].priority
        )
        
        return enabled
    
    def get_algorithm_config(self, name: str) -> Optional[AlgorithmConfig]:
        """获取算法配置"""
        return self.ALGORITHMS.get(name)
    
    def is_algorithm_enabled(self, name: str) -> bool:
        """检查算法是否在当前模式下启用"""
        config = self.ALGORITHMS.get(name)
        if not config:
            return False
        
        if self._detected_mode == ExecutionMode.LOCAL:
            return config.enabled_in_local
        else:
            return config.enabled_in_api
    
    def initialize_algorithms(self) -> Dict[str, Any]:
        """
        初始化所有启用的算法
        
        Returns:
            算法实例字典
        """
        active = self.get_active_algorithms()
        instances = {}
        
        logger.info(f"正在初始化 {len(active)} 个算法...")
        
        for algo_name in active:
            config = self.ALGORITHMS[algo_name]
            try:
                # 动态导入模块
                module = __import__(
                    config.module_path,
                    fromlist=[config.class_name]
                )
                cls = getattr(module, config.class_name)
                
                # 实例化
                instance = cls()
                instances[algo_name] = instance
                logger.debug(f"✓ 初始化成功: {config.name}")
                
            except Exception as e:
                logger.warning(f"✗ 初始化失败: {config.name} - {e}")
                continue
        
        self._algorithm_instances = instances
        logger.info(f"成功初始化 {len(instances)}/{len(active)} 个算法")
        
        return instances
    
    def get_algorithm(self, name: str) -> Optional[Any]:
        """获取算法实例"""
        return self._algorithm_instances.get(name)
    
    def get_mode_info(self) -> Dict[str, Any]:
        """获取当前模式信息"""
        active = self.get_active_algorithms()
        
        return {
            "mode": self._detected_mode.name,
            "total_algorithms": len(self.ALGORITHMS),
            "active_algorithms": len(active),
            "api_algorithms": sum(1 for c in self.ALGORITHMS.values() if c.enabled_in_api),
            "local_algorithms": sum(1 for c in self.ALGORITHMS.values() if c.enabled_in_local),
            "active_list": [self.ALGORITHMS[a].name for a in active]
        }
    
    def switch_mode(self, mode: ExecutionMode) -> None:
        """
        手动切换模式
        
        Args:
            mode: 新执行模式
        """
        if mode == ExecutionMode.AUTO:
            self._detected_mode = self._detect_execution_mode()
        else:
            self._detected_mode = mode
        
        # 清空已初始化的实例
        self._algorithm_instances.clear()
        
        logger.info(f"已切换到 {self._detected_mode.name} 模式")


# 全局选择器实例
_selector: Optional[AdaptiveAlgorithmSelector] = None


def get_selector(mode: ExecutionMode = ExecutionMode.AUTO) -> AdaptiveAlgorithmSelector:
    """
    获取全局选择器实例（单例模式）
    
    Args:
        mode: 执行模式
        
    Returns:
        AdaptiveAlgorithmSelector实例
    """
    global _selector
    if _selector is None:
        _selector = AdaptiveAlgorithmSelector(mode)
    return _selector


def reset_selector() -> None:
    """重置选择器（用于测试）"""
    global _selector
    _selector = None


# 便捷函数
def get_active_algorithms() -> List[str]:
    """获取当前启用的算法列表"""
    return get_selector().get_active_algorithms()


def is_local_mode() -> bool:
    """检查是否为本地模式"""
    selector = get_selector()
    return selector._detected_mode == ExecutionMode.LOCAL


def is_api_mode() -> bool:
    """检查是否为API模式"""
    selector = get_selector()
    return selector._detected_mode == ExecutionMode.API


# 使用示例
if __name__ == "__main__":
    # 测试自适应选择器
    selector = AdaptiveAlgorithmSelector(ExecutionMode.AUTO)
    
    info = selector.get_mode_info()
    print("=" * 60)
    print(f"执行模式: {info['mode']}")
    print(f"总算法数: {info['total_algorithms']}")
    print(f"当前启用: {info['active_algorithms']}")
    print("=" * 60)
    print("\n启用的算法:")
    for i, name in enumerate(info['active_list'], 1):
        print(f"  {i}. {name}")
