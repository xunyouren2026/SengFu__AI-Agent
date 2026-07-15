"""
模型注册中心 (Model Registry)

管理所有已注册的模型配置，提供统一的模型发现、检索和推荐服务。

核心功能:
- 模型注册与注销
- 模型查询与搜索
- 按能力过滤
- 智能推荐
- 提供商统计
- 配置导入导出
- 预置配置管理

Author: AGI Team
Version: 1.0.0
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import (
    Dict, List, Optional, Any, Set, Tuple,
    Callable, Union
)
from copy import deepcopy

from .universal import UniversalModelConfig, UniversalLLMProvider
from .model_config import (
    ModelConfigManager,
    ConfigValidator,
    ConfigValidationResult,
)

logger = logging.getLogger(__name__)


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class ModelInfo:
    """
    模型信息

    包含模型配置和运行时状态。

    Attributes:
        config: 模型配置
        registered_at: 注册时间
        last_used_at: 最后使用时间
        is_enabled: 是否启用
        is_healthy: 是否健康
        health_check_failures: 健康检查失败次数
        tags: 标签
        description: 描述
    """
    config: UniversalModelConfig
    registered_at: datetime = field(default_factory=datetime.now)
    last_used_at: Optional[datetime] = None
    is_enabled: bool = True
    is_healthy: bool = True
    health_check_failures: int = 0
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "config": self.config.to_dict(),
            "registered_at": self.registered_at.isoformat(),
            "last_used_at": (
                self.last_used_at.isoformat() if self.last_used_at else None
            ),
            "is_enabled": self.is_enabled,
            "is_healthy": self.is_healthy,
            "health_check_failures": self.health_check_failures,
            "tags": self.tags,
            "description": self.description,
        }


@dataclass
class ProviderStats:
    """
    提供商统计信息

    Attributes:
        provider: 提供商名称
        display_name: 显示名称
        model_count: 模型数量
        enabled_count: 启用的模型数量
        healthy_count: 健康的模型数量
        avg_max_context: 平均最大上下文
        supports_vision: 是否有支持视觉的模型
        supports_function_call: 是否有支持函数调用的模型
        supports_stream: 是否有支持流式的模型
        min_input_price: 最低输入价格
        min_output_price: 最低输出价格
    """
    provider: str = ""
    display_name: str = ""
    model_count: int = 0
    enabled_count: int = 0
    healthy_count: int = 0
    avg_max_context: float = 0.0
    supports_vision: bool = False
    supports_function_call: bool = False
    supports_stream: bool = False
    min_input_price: float = 0.0
    min_output_price: float = 0.0


@dataclass
class RecommendationCriteria:
    """
    推荐标准

    Attributes:
        task_type: 任务类型
        language: 语言 (zh/en)
        need_vision: 是否需要视觉能力
        need_function_call: 是否需要函数调用
        need_stream: 是否需要流式
        max_cost: 最大预算
        min_context: 最小上下文长度
        preferred_providers: 偏好的提供商
        excluded_providers: 排除的提供商
    """
    task_type: str = "general"  # general, coding, writing, translation, analysis, creative
    language: str = "zh"
    need_vision: bool = False
    need_function_call: bool = False
    need_stream: bool = True
    max_cost: float = 0.0  # 0表示不限
    min_context: int = 0
    preferred_providers: List[str] = field(default_factory=list)
    excluded_providers: List[str] = field(default_factory=list)


# ============================================================
# 模型注册中心
# ============================================================

class ModelRegistry:
    """
    模型注册中心

    管理所有已注册的模型配置，提供统一的模型发现、检索和推荐服务。

    Features:
    - 模型注册与注销
    - 模型查询与搜索
    - 按能力过滤
    - 智能推荐
    - 提供商统计
    - 配置导入导出
    - 预置配置管理

    Example:
        ```python
        # 创建注册中心
        registry = ModelRegistry()

        # 注册模型
        registry.register_model(UniversalModelConfig(
            model_id="zhipu/glm-4",
            model_name="GLM-4",
            provider="zhipu",
            api_base="https://open.bigmodel.cn/api/paas/v4",
            api_key="your-key",
        ))

        # 搜索模型
        models = registry.search_models("GLM")

        # 获取推荐
        recommended = registry.get_recommended_model(
            task_type="coding", language="zh"
        )

        # 列出所有提供商
        providers = registry.list_providers()
        ```
    """

    # 内置的预置配置目录
    PRESETS_DIR = Path(__file__).parent / "presets"

    def __init__(
        self,
        config_manager: Optional[ModelConfigManager] = None,
        auto_health_check: bool = False,
    ):
        """
        初始化模型注册中心。

        Args:
            config_manager: 配置管理器 (可选)
            auto_health_check: 是否自动健康检查
        """
        self._models: Dict[str, ModelInfo] = {}
        self._config_manager = config_manager or ModelConfigManager()
        self._auto_health_check = auto_health_check
        self._health_check_task: Optional[asyncio.Task] = None
        self._change_callbacks: List[Callable] = []

        # 任务类型到推荐模型的映射
        self._task_recommendations: Dict[str, List[str]] = {
            "general": [
                "deepseek/deepseek-chat",
                "qwen/qwen-max",
                "zhipu/glm-4",
                "kimi/moonshot-v1-128k",
            ],
            "coding": [
                "deepseek/deepseek-coder",
                "qwen/qwen-coder-plus",
                "zhipu/glm-4",
            ],
            "writing": [
                "kimi/moonshot-v1-128k",
                "qwen/qwen-max",
                "minimax/abab6.5-chat",
            ],
            "translation": [
                "deepseek/deepseek-chat",
                "qwen/qwen-max",
                "yi/yi-large",
            ],
            "analysis": [
                "qwen/qwen-max",
                "zhipu/glm-4-plus",
                "deepseek/deepseek-chat",
            ],
            "creative": [
                "kimi/moonshot-v1-128k",
                "minimax/abab6.5-chat",
                "baichuan/baichuan4",
            ],
            "math": [
                "zhipu/glm-4-plus",
                "deepseek/deepseek-chat",
                "qwen/qwen-max",
            ],
            "vision": [
                "qwen/qwen-vl-max",
                "zhipu/glm-4v",
                "minimax/abab6.5-chat",
            ],
        }

    # ========================================================
    # 模型注册与注销
    # ========================================================

    def register_model(
        self,
        config: UniversalModelConfig,
        tags: Optional[List[str]] = None,
        description: str = "",
        validate: bool = True,
    ) -> bool:
        """
        注册模型

        Args:
            config: 模型配置
            tags: 标签列表
            description: 模型描述
            validate: 是否验证配置

        Returns:
            是否注册成功
        """
        # 验证配置
        if validate:
            result = self._config_manager.validate_model(config.to_dict())
            if not result.valid:
                logger.error(
                    f"模型配置验证失败 ({config.model_id}): "
                    f"{result.errors}"
                )
                return False
            if result.warnings:
                logger.warning(
                    f"模型配置警告 ({config.model_id}): "
                    f"{result.warnings}"
                )

        model_id = config.model_id
        is_update = model_id in self._models

        self._models[model_id] = ModelInfo(
            config=config,
            tags=tags or [],
            description=description,
        )

        action = "更新" if is_update else "注册"
        logger.info(f"模型已{action}: {model_id}")

        # 触发回调
        self._notify_change("registered" if not is_update else "updated", model_id)

        return True

    def register_models(
        self,
        configs: List[UniversalModelConfig],
        validate: bool = True,
    ) -> Tuple[int, int]:
        """
        批量注册模型

        Args:
            configs: 模型配置列表
            validate: 是否验证配置

        Returns:
            (成功数, 失败数)
        """
        success = 0
        failed = 0

        for config in configs:
            if self.register_model(config, validate=validate):
                success += 1
            else:
                failed += 1

        logger.info(f"批量注册完成: 成功 {success}, 失败 {failed}")
        return success, failed

    def unregister_model(self, model_id: str) -> bool:
        """
        注销模型

        Args:
            model_id: 模型ID

        Returns:
            是否成功
        """
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"模型已注销: {model_id}")
            self._notify_change("unregistered", model_id)
            return True
        return False

    def enable_model(self, model_id: str) -> bool:
        """
        启用模型

        Args:
            model_id: 模型ID

        Returns:
            是否成功
        """
        info = self._models.get(model_id)
        if info:
            info.is_enabled = True
            logger.info(f"模型已启用: {model_id}")
            return True
        return False

    def disable_model(self, model_id: str) -> bool:
        """
        禁用模型

        Args:
            model_id: 模型ID

        Returns:
            是否成功
        """
        info = self._models.get(model_id)
        if info:
            info.is_enabled = False
            logger.info(f"模型已禁用: {model_id}")
            return True
        return False

    # ========================================================
    # 模型查询
    # ========================================================

    def get_model(self, model_id: str) -> Optional[UniversalModelConfig]:
        """
        获取模型配置

        Args:
            model_id: 模型ID

        Returns:
            模型配置，未找到返回None
        """
        info = self._models.get(model_id)
        if info and info.is_enabled:
            return info.config
        return None

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        获取模型完整信息

        Args:
            model_id: 模型ID

        Returns:
            模型信息，未找到返回None
        """
        return self._models.get(model_id)

    def get_model_or_raise(self, model_id: str) -> UniversalModelConfig:
        """
        获取模型配置，未找到时抛出异常

        Args:
            model_id: 模型ID

        Returns:
            模型配置

        Raises:
            KeyError: 模型不存在
        """
        config = self.get_model(model_id)
        if config is None:
            available = list(self._models.keys())
            raise KeyError(
                f"模型不存在或已禁用: {model_id}。"
                f"可用模型: {available}"
            )
        return config

    def list_models(
        self,
        provider: Optional[str] = None,
        enabled_only: bool = True,
        healthy_only: bool = False,
    ) -> List[UniversalModelConfig]:
        """
        列出所有模型

        Args:
            provider: 按提供商过滤
            enabled_only: 仅显示启用的模型
            healthy_only: 仅显示健康的模型

        Returns:
            模型配置列表
        """
        models: List[UniversalModelConfig] = []

        for info in self._models.values():
            if enabled_only and not info.is_enabled:
                continue
            if healthy_only and not info.is_healthy:
                continue
            if provider and info.config.provider != provider:
                continue
            models.append(info.config)

        # 按提供商和模型名排序
        models.sort(key=lambda m: (m.provider, m.model_name))
        return models

    def list_providers(self) -> List[Dict[str, Any]]:
        """
        列出所有提供商

        Returns:
            提供商信息列表
        """
        providers: Dict[str, Dict[str, Any]] = {}

        for info in self._models.values():
            provider = info.config.provider
            if provider not in providers:
                providers[provider] = {
                    "name": provider,
                    "model_count": 0,
                    "enabled_count": 0,
                    "models": [],
                }

            providers[provider]["model_count"] += 1
            if info.is_enabled:
                providers[provider]["enabled_count"] += 1
            providers[provider]["models"].append(info.config.model_id)

        return sorted(providers.values(), key=lambda p: p["name"])

    # ========================================================
    # 模型搜索与过滤
    # ========================================================

    def search_models(
        self,
        query: str,
        limit: int = 20
    ) -> List[UniversalModelConfig]:
        """
        搜索模型

        支持按模型ID、名称、提供商进行模糊搜索。

        Args:
            query: 搜索关键词
            limit: 最大返回数量

        Returns:
            匹配的模型配置列表
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return self.list_models()[:limit]

        scored_models: List[Tuple[float, UniversalModelConfig]] = []

        for info in self._models.values():
            if not info.is_enabled:
                continue

            config = info.config
            score = 0.0

            # 精确匹配模型ID
            if query_lower == config.model_id.lower():
                score = 100.0
            elif query_lower in config.model_id.lower():
                score = 80.0

            # 精确匹配模型名
            if query_lower == config.model_name.lower():
                score = max(score, 90.0)
            elif query_lower in config.model_name.lower():
                score = max(score, 70.0)

            # 匹配提供商
            if query_lower == config.provider.lower():
                score = max(score, 60.0)
            elif query_lower in config.provider.lower():
                score = max(score, 40.0)

            # 匹配标签
            for tag in info.tags:
                if query_lower in tag.lower():
                    score = max(score, 50.0)

            # 匹配描述
            if info.description and query_lower in info.description.lower():
                score = max(score, 30.0)

            if score > 0:
                scored_models.append((score, config))

        # 按分数排序
        scored_models.sort(key=lambda x: (-x[0], x[1].model_name))

        return [m for _, m in scored_models[:limit]]

    def filter_models(
        self,
        capabilities: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        min_context: int = 0,
        max_cost_per_1k: float = 0.0,
        protocol: Optional[str] = None,
        auth_type: Optional[str] = None,
    ) -> List[UniversalModelConfig]:
        """
        按条件过滤模型

        Args:
            capabilities: 能力要求，如:
                {"supports_vision": True, "supports_function_call": True}
            provider: 提供商
            min_context: 最小上下文长度
            max_cost_per_1k: 最大每千token成本 (0表示不限)
            protocol: 协议类型
            auth_type: 认证类型

        Returns:
            匹配的模型配置列表
        """
        results: List[UniversalModelConfig] = []

        for info in self._models.values():
            if not info.is_enabled:
                continue

            config = info.config
            match = True

            # 能力过滤
            if capabilities:
                for cap_key, cap_value in capabilities.items():
                    if hasattr(config, cap_key):
                        if getattr(config, cap_key) != cap_value:
                            match = False
                            break

            # 提供商过滤
            if provider and config.provider != provider:
                match = False

            # 上下文长度过滤
            if min_context > 0 and config.max_context < min_context:
                match = False

            # 成本过滤
            if max_cost_per_1k > 0:
                total_price = (
                    config.input_price_per_1k + config.output_price_per_1k
                )
                if total_price > max_cost_per_1k:
                    match = False

            # 协议过滤
            if protocol and config.api_protocol != protocol:
                match = False

            # 认证类型过滤
            if auth_type and config.auth_type != auth_type:
                match = False

            if match:
                results.append(config)

        return results

    # ========================================================
    # 模型推荐
    # ========================================================

    def get_recommended_model(
        self,
        task_type: str = "general",
        language: str = "zh",
        criteria: Optional[RecommendationCriteria] = None,
    ) -> Optional[UniversalModelConfig]:
        """
        获取推荐模型

        根据任务类型、语言和其他条件推荐最合适的模型。

        Args:
            task_type: 任务类型
            language: 语言
            criteria: 额外推荐标准

        Returns:
            推荐的模型配置
        """
        if criteria is None:
            criteria = RecommendationCriteria(
                task_type=task_type,
                language=language,
            )

        # 获取候选模型列表
        candidates = self._task_recommendations.get(
            criteria.task_type, self._task_recommendations["general"]
        )

        # 按优先级尝试
        for model_id in candidates:
            info = self._models.get(model_id)
            if not info or not info.is_enabled or not info.is_healthy:
                continue

            config = info.config

            # 检查能力要求
            if criteria.need_vision and not config.supports_vision:
                continue
            if criteria.need_function_call and not config.supports_function_call:
                continue
            if criteria.need_stream and not config.supports_stream:
                continue
            if criteria.min_context > 0 and config.max_context < criteria.min_context:
                continue

            # 检查排除的提供商
            if config.provider in criteria.excluded_providers:
                continue

            # 检查成本
            if criteria.max_cost > 0:
                cost = self._estimate_single_cost(config, 1000, 1000)
                if cost > criteria.max_cost:
                    continue

            # 更新最后使用时间
            info.last_used_at = datetime.now()

            return config

        # 如果推荐列表中没有合适的，从所有模型中选择
        logger.info(
            f"推荐列表中无合适模型，从全部模型中搜索 "
            f"(task={criteria.task_type})"
        )

        all_models = self.filter_models(
            capabilities={
                "supports_vision": criteria.need_vision,
                "supports_function_call": criteria.need_function_call,
                "supports_stream": criteria.need_stream,
            },
            min_context=criteria.min_context,
        )

        if all_models:
            # 优先选择偏好提供商
            for model in all_models:
                if model.provider in criteria.preferred_providers:
                    return model
            return all_models[0]

        return None

    def get_recommended_models(
        self,
        task_type: str = "general",
        language: str = "zh",
        limit: int = 5,
    ) -> List[UniversalModelConfig]:
        """
        获取推荐模型列表

        Args:
            task_type: 任务类型
            language: 语言
            limit: 最大返回数量

        Returns:
            推荐的模型配置列表
        """
        criteria = RecommendationCriteria(
            task_type=task_type,
            language=language,
        )

        results: List[UniversalModelConfig] = []
        seen_providers: Set[str] = set()

        # 从推荐列表获取
        candidates = self._task_recommendations.get(
            task_type, self._task_recommendations["general"]
        )

        for model_id in candidates:
            if len(results) >= limit:
                break

            info = self._models.get(model_id)
            if not info or not info.is_enabled:
                continue

            # 避免同一提供商重复
            if info.config.provider in seen_providers:
                continue

            seen_providers.add(info.config.provider)
            results.append(info.config)

        # 补充其他模型
        if len(results) < limit:
            for info in self._models.values():
                if len(results) >= limit:
                    break
                if not info.is_enabled:
                    continue
                if info.config.provider in seen_providers:
                    continue
                seen_providers.add(info.config.provider)
                results.append(info.config)

        return results

    @staticmethod
    def _estimate_single_cost(
        config: UniversalModelConfig,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """估算单个模型的调用成本"""
        return (
            (input_tokens / 1000.0) * config.input_price_per_1k
            + (output_tokens / 1000.0) * config.output_price_per_1k
        )

    # ========================================================
    # 配置导入导出
    # ========================================================

    def load_from_config(
        self,
        config_path: str,
        process_env: bool = True,
        validate: bool = True,
    ) -> Tuple[int, int]:
        """
        从配置文件加载模型

        Args:
            config_path: 配置文件路径
            process_env: 是否处理环境变量
            validate: 是否验证配置

        Returns:
            (成功数, 失败数)
        """
        data = self._config_manager.load_from_file(
            config_path, process_env=process_env
        )

        return self._load_from_data(data, validate=validate)

    def load_from_dict(
        self,
        data: Dict[str, Any],
        validate: bool = True,
    ) -> Tuple[int, int]:
        """
        从字典加载模型

        Args:
            data: 配置数据
            validate: 是否验证

        Returns:
            (成功数, 失败数)
        """
        return self._load_from_data(data, validate=validate)

    def _load_from_data(
        self,
        data: Dict[str, Any],
        validate: bool = True,
    ) -> Tuple[int, int]:
        """
        从配置数据加载模型

        Args:
            data: 配置数据
            validate: 是否验证

        Returns:
            (成功数, 失败数)
        """
        providers = data.get("providers", {})
        success = 0
        failed = 0

        for provider_name, provider_data in providers.items():
            if not isinstance(provider_data, dict):
                continue

            display_name = provider_data.get("name", provider_name)
            models = provider_data.get("models", [])

            for model_data in models:
                if not isinstance(model_data, dict):
                    continue

                try:
                    # 确保provider字段正确
                    model_data["provider"] = provider_name

                    config = UniversalModelConfig.from_dict(model_data)

                    if self.register_model(
                        config,
                        tags=model_data.get("tags", []),
                        description=model_data.get("description", ""),
                        validate=validate,
                    ):
                        success += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.error(
                        f"加载模型失败 ({model_data.get('model_id', 'unknown')}): {e}"
                    )
                    failed += 1

        logger.info(
            f"配置加载完成: 成功 {success}, 失败 {failed}"
        )
        return success, failed

    def save_to_config(
        self,
        config_path: str,
        format: str = "yaml",
        encrypt_keys: bool = False,
    ) -> None:
        """
        保存当前所有模型配置到文件

        Args:
            config_path: 保存路径
            format: 格式 (yaml/json)
            encrypt_keys: 是否加密API密钥
        """
        data = self.export_config()

        self._config_manager.save_to_file(
            data, config_path, format=format, encrypt_keys=encrypt_keys
        )

    def export_config(self) -> Dict[str, Any]:
        """
        导出当前所有模型配置

        Returns:
            配置字典
        """
        providers: Dict[str, Any] = {}

        for info in self._models.values():
            provider = info.config.provider
            if provider not in providers:
                providers[provider] = {
                    "name": provider,
                    "models": [],
                }

            model_data = info.config.to_dict()
            if info.tags:
                model_data["tags"] = info.tags
            if info.description:
                model_data["description"] = info.description

            providers[provider]["models"].append(model_data)

        return {"providers": providers}

    # ========================================================
    # 预置配置
    # ========================================================

    def import_preset(
        self,
        preset_name: str,
        validate: bool = True,
    ) -> Tuple[int, int]:
        """
        导入预置配置

        Args:
            preset_name: 预置配置名称 (如 "china_models")
            validate: 是否验证

        Returns:
            (成功数, 失败数)
        """
        # 查找预置配置文件
        preset_path = self.PRESETS_DIR / f"{preset_name}.yaml"

        if not preset_path.exists():
            preset_path = self.PRESETS_DIR / f"{preset_name}.json"

        if not preset_path.exists():
            logger.error(f"预置配置不存在: {preset_name}")
            return 0, 0

        logger.info(f"正在导入预置配置: {preset_name}")
        return self.load_from_config(
            str(preset_path),
            process_env=True,
            validate=validate,
        )

    def list_presets(self) -> List[Dict[str, str]]:
        """
        列出可用的预置配置

        Returns:
            预置配置信息列表
        """
        presets: List[Dict[str, str]] = []

        if not self.PRESETS_DIR.exists():
            return presets

        for file_path in self.PRESETS_DIR.glob("*.yaml"):
            presets.append({
                "name": file_path.stem,
                "path": str(file_path),
                "format": "yaml",
            })

        for file_path in self.PRESETS_DIR.glob("*.json"):
            presets.append({
                "name": file_path.stem,
                "path": str(file_path),
                "format": "json",
            })

        return sorted(presets, key=lambda p: p["name"])

    # ========================================================
    # 模型验证
    # ========================================================

    def validate_model(self, model_id: str) -> ConfigValidationResult:
        """
        验证指定模型的配置

        Args:
            model_id: 模型ID

        Returns:
            验证结果
        """
        info = self._models.get(model_id)
        if not info:
            result = ConfigValidationResult(model_id=model_id)
            result.add_error(f"模型不存在: {model_id}")
            return result

        return self._config_manager.validate_model(info.config.to_dict())

    def validate_all_models(self) -> Dict[str, ConfigValidationResult]:
        """
        验证所有已注册模型的配置

        Returns:
            模型ID到验证结果的映射
        """
        results: Dict[str, ConfigValidationResult] = {}
        for model_id in self._models:
            results[model_id] = self.validate_model(model_id)
        return results

    # ========================================================
    # 提供商统计
    # ========================================================

    def get_provider_stats(self) -> Dict[str, ProviderStats]:
        """
        获取所有提供商的统计信息

        Returns:
            提供商名称到统计信息的映射
        """
        stats: Dict[str, ProviderStats] = {}

        for info in self._models.values():
            provider = info.config.provider
            config = info.config

            if provider not in stats:
                stats[provider] = ProviderStats(
                    provider=provider,
                    display_name=provider,
                    min_input_price=float("inf"),
                    min_output_price=float("inf"),
                )

            stat = stats[provider]
            stat.model_count += 1

            if info.is_enabled:
                stat.enabled_count += 1
            if info.is_healthy:
                stat.healthy_count += 1

            stat.avg_max_context += config.max_context

            if config.supports_vision:
                stat.supports_vision = True
            if config.supports_function_call:
                stat.supports_function_call = True
            if config.supports_stream:
                stat.supports_stream = True

            if config.input_price_per_1k < stat.min_input_price:
                stat.min_input_price = config.input_price_per_1k
            if config.output_price_per_1k < stat.min_output_price:
                stat.min_output_price = config.output_price_per_1k

        # 计算平均值
        for stat in stats.values():
            if stat.model_count > 0:
                stat.avg_max_context /= stat.model_count
            if stat.min_input_price == float("inf"):
                stat.min_input_price = 0.0
            if stat.min_output_price == float("inf"):
                stat.min_output_price = 0.0

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """
        获取注册中心整体统计

        Returns:
            统计信息字典
        """
        total = len(self._models)
        enabled = sum(1 for i in self._models.values() if i.is_enabled)
        healthy = sum(1 for i in self._models.values() if i.is_healthy)
        providers = len(set(i.config.provider for i in self._models.values()))

        return {
            "total_models": total,
            "enabled_models": enabled,
            "healthy_models": healthy,
            "disabled_models": total - enabled,
            "unhealthy_models": total - healthy,
            "total_providers": providers,
            "providers": self.list_providers(),
        }

    # ========================================================
    # Provider创建
    # ========================================================

    def create_provider(
        self,
        model_id: str,
        **kwargs: Any
    ) -> UniversalLLMProvider:
        """
        为指定模型创建UniversalLLMProvider实例

        Args:
            model_id: 模型ID
            **kwargs: 覆盖配置参数

        Returns:
            UniversalLLMProvider实例

        Raises:
            KeyError: 模型不存在
        """
        config = self.get_model_or_raise(model_id)

        # 应用覆盖参数
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = UniversalModelConfig.from_dict(config_dict)

        return UniversalLLMProvider(model_config=config)

    def create_provider_for_task(
        self,
        task_type: str = "general",
        language: str = "zh",
        **kwargs: Any
    ) -> UniversalLLMProvider:
        """
        为指定任务创建最合适的Provider

        Args:
            task_type: 任务类型
            language: 语言
            **kwargs: 覆盖配置参数

        Returns:
            UniversalLLMProvider实例

        Raises:
            RuntimeError: 没有可用的模型
        """
        config = self.get_recommended_model(
            task_type=task_type,
            language=language,
        )

        if not config:
            raise RuntimeError(
                f"没有可用的模型 (task={task_type}, lang={language})"
            )

        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = UniversalModelConfig.from_dict(config_dict)

        return UniversalLLMProvider(model_config=config)

    # ========================================================
    # 健康检查
    # ========================================================

    async def check_health(
        self,
        model_id: str,
        timeout: float = 10.0,
    ) -> bool:
        """
        检查模型健康状态

        Args:
            model_id: 模型ID
            timeout: 超时时间

        Returns:
            是否健康
        """
        info = self._models.get(model_id)
        if not info:
            return False

        try:
            provider = self.create_provider(model_id)
            provider.initialize()

            result = await asyncio.wait_for(
                provider.test_connection(),
                timeout=timeout,
            )

            if result["success"]:
                info.is_healthy = True
                info.health_check_failures = 0
                return True
            else:
                info.health_check_failures += 1
                if info.health_check_failures >= 3:
                    info.is_healthy = False
                return False

        except asyncio.TimeoutError:
            info.health_check_failures += 1
            if info.health_check_failures >= 3:
                info.is_healthy = False
            logger.warning(f"健康检查超时: {model_id}")
            return False
        except Exception as e:
            info.health_check_failures += 1
            if info.health_check_failures >= 3:
                info.is_healthy = False
            logger.error(f"健康检查失败 ({model_id}): {e}")
            return False

    async def check_all_health(self) -> Dict[str, bool]:
        """
        检查所有已启用模型的健康状态

        Returns:
            模型ID到健康状态的映射
        """
        results: Dict[str, bool] = {}

        tasks = []
        model_ids = []

        for model_id, info in self._models.items():
            if info.is_enabled:
                tasks.append(self.check_health(model_id))
                model_ids.append(model_id)

        if tasks:
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            for mid, result in zip(model_ids, health_results):
                if isinstance(result, Exception):
                    results[mid] = False
                else:
                    results[mid] = result

        return results

    # ========================================================
    # 变更通知
    # ========================================================

    def on_change(
        self,
        callback: Callable[[str, str], None]
    ) -> None:
        """
        注册变更回调

        Args:
            callback: 回调函数 (change_type, model_id)
        """
        self._change_callbacks.append(callback)

    def _notify_change(self, change_type: str, model_id: str) -> None:
        """通知变更"""
        for callback in self._change_callbacks:
            try:
                callback(change_type, model_id)
            except Exception as e:
                logger.error(f"变更回调执行失败: {e}")

    # ========================================================
    # 生命周期
    # ========================================================

    async def start(self) -> None:
        """启动注册中心"""
        if self._auto_health_check:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )
            logger.info("自动健康检查已启动")

    async def stop(self) -> None:
        """停止注册中心"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        await self._config_manager.stop_all_watching()
        logger.info("注册中心已停止")

    async def _health_check_loop(self) -> None:
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次
                await self.check_all_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")

    def clear(self) -> None:
        """清空所有已注册的模型"""
        self._models.clear()
        logger.info("已清空所有注册模型")
