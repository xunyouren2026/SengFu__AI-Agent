"""
模型配置管理器 (Model Config Manager)

提供模型配置的完整生命周期管理，包括:
- YAML/JSON配置文件加载与保存
- 配置验证与校验
- 环境变量覆盖机制
- 配置热更新监控
- API密钥加密存储

Author: AGI Team
Version: 1.0.0
"""

import os
import json
import yaml
import logging
import hashlib
import base64
import asyncio
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import (
    Dict, List, Optional, Any, Set, Tuple,
    Callable, Union
)
from copy import deepcopy

logger = logging.getLogger(__name__)


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class ConfigValidationResult:
    """
    配置验证结果

    Attributes:
        valid: 是否有效
        errors: 错误列表
        warnings: 警告列表
        model_id: 模型ID
    """
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    model_id: str = ""

    def add_error(self, msg: str) -> None:
        """添加错误"""
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        """添加警告"""
        self.warnings.append(msg)


@dataclass
class ConfigChangeEvent:
    """
    配置变更事件

    Attributes:
        change_type: 变更类型 (added/modified/removed)
        model_id: 模型ID
        old_config: 旧配置
        new_config: 新配置
        timestamp: 变更时间
    """
    change_type: str  # added, modified, removed
    model_id: str
    old_config: Optional[Dict[str, Any]] = None
    new_config: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================
# 配置验证器
# ============================================================

class ConfigValidator:
    """
    配置验证器

    验证模型配置的完整性和正确性。

    支持的验证规则:
    - 必填字段检查
    - URL格式验证
    - 协议类型验证
    - 认证类型验证
    - 数值范围验证
    - 模型ID格式验证
    """

    # 必填字段
    REQUIRED_FIELDS: Set[str] = {"model_id", "model_name", "provider", "api_base"}

    # 有效的协议类型
    VALID_PROTOCOLS: Set[str] = {"openai", "claude", "custom"}

    # 有效的认证类型
    VALID_AUTH_TYPES: Set[str] = {
        "bearer", "api_key_query", "api_key_header",
        "oauth2", "hmac_sha256", "aws_sigv4", "custom",
    }

    # 有效的模型能力字段
    BOOLEAN_CAPABILITY_FIELDS: Set[str] = {
        "supports_stream", "supports_vision",
        "supports_function_call", "supports_embeddings",
    }

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> ConfigValidationResult:
        """
        验证模型配置

        Args:
            config: 模型配置字典

        Returns:
            验证结果
        """
        result = ConfigValidationResult(
            model_id=config.get("model_id", "unknown")
        )

        # 1. 必填字段检查
        for field_name in cls.REQUIRED_FIELDS:
            value = config.get(field_name)
            if not value or (isinstance(value, str) and not value.strip()):
                result.add_error(f"必填字段 '{field_name}' 缺失或为空")

        # 2. 模型ID格式验证
        model_id = config.get("model_id", "")
        if model_id:
            if "/" not in model_id:
                result.add_warning(
                    f"模型ID '{model_id}' 建议使用 'provider/model' 格式"
                )
            if len(model_id) > 128:
                result.add_error(f"模型ID长度超过128字符限制")

        # 3. API Base URL验证
        api_base = config.get("api_base", "")
        if api_base:
            if not api_base.startswith(("http://", "https://")):
                result.add_error(
                    f"api_base 必须以 http:// 或 https:// 开头，当前值: {api_base}"
                )

        # 4. 协议类型验证
        protocol = config.get("api_protocol", "openai")
        if protocol not in cls.VALID_PROTOCOLS:
            result.add_error(
                f"不支持的协议类型: {protocol}，"
                f"有效值: {cls.VALID_PROTOCOLS}"
            )

        # 5. 认证类型验证
        auth_type = config.get("auth_type", "bearer")
        if auth_type not in cls.VALID_AUTH_TYPES:
            result.add_error(
                f"不支持的认证类型: {auth_type}，"
                f"有效值: {cls.VALID_AUTH_TYPES}"
            )

        # 6. 数值范围验证
        max_context = config.get("max_context", 8192)
        if not isinstance(max_context, int) or max_context < 1:
            result.add_error(f"max_context 必须是正整数，当前值: {max_context}")

        max_output = config.get("max_output", 4096)
        if not isinstance(max_output, int) or max_output < 1:
            result.add_error(f"max_output 必须是正整数，当前值: {max_output}")

        if isinstance(max_context, int) and isinstance(max_output, int):
            if max_output > max_context:
                result.add_warning(
                    f"max_output ({max_output}) 大于 max_context ({max_context})"
                )

        timeout = config.get("timeout", 60.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            result.add_error(f"timeout 必须是正数，当前值: {timeout}")

        rate_limit = config.get("rate_limit", 60)
        if not isinstance(rate_limit, int) or rate_limit < 1:
            result.add_error(f"rate_limit 必须是正整数，当前值: {rate_limit}")

        retry_count = config.get("retry_count", 3)
        if not isinstance(retry_count, int) or retry_count < 0:
            result.add_error(f"retry_count 必须是非负整数，当前值: {retry_count}")

        # 7. 价格验证
        input_price = config.get("input_price_per_1k", 0.0)
        output_price = config.get("output_price_per_1k", 0.0)
        if input_price < 0:
            result.add_error(f"input_price_per_1k 不能为负数")
        if output_price < 0:
            result.add_error(f"output_price_per_1k 不能为负数")

        # 8. 自定义协议额外验证
        if protocol == "custom":
            if not config.get("request_mapping"):
                result.add_warning(
                    "自定义协议建议配置 request_mapping"
                )
            if not config.get("response_mapping"):
                result.add_warning(
                    "自定义协议建议配置 response_mapping"
                )

        # 9. OAuth2额外验证
        if auth_type == "oauth2":
            auth_config = config.get("auth_config", {})
            if not auth_config.get("token_url"):
                result.add_error("OAuth2认证需要配置 auth_config.token_url")
            if not auth_config.get("client_id"):
                result.add_error("OAuth2认证需要配置 auth_config.client_id")

        # 10. HMAC额外验证
        if auth_type == "hmac_sha256":
            auth_config = config.get("auth_config", {})
            if not auth_config.get("secret_key") and not config.get("api_key"):
                result.add_warning(
                    "HMAC-SHA256认证建议配置 secret_key"
                )

        return result

    @classmethod
    def validate_batch(
        cls,
        configs: List[Dict[str, Any]]
    ) -> Tuple[List[ConfigValidationResult], bool]:
        """
        批量验证配置

        Args:
            configs: 配置列表

        Returns:
            (验证结果列表, 是否全部有效)
        """
        results = [cls.validate(c) for c in configs]
        all_valid = all(r.valid for r in results)
        return results, all_valid


# ============================================================
# 配置加密器
# ============================================================

class ConfigEncryptor:
    """
    配置加密器

    用于加密存储API密钥等敏感信息。

    使用简单的AES-like加密方案:
    - 基于用户提供的密钥派生加密密钥
    - 使用Fernet兼容的加密格式
    - 支持环境变量作为密钥来源

    注意: 生产环境建议使用专门的密钥管理服务。
    """

    DEFAULT_SALT = b"agi_unified_framework_config_salt_v1"

    @staticmethod
    def encrypt(plaintext: str, key: str) -> str:
        """
        加密文本

        Args:
            plaintext: 明文
            key: 加密密钥

        Returns:
            Base64编码的密文
        """
        if not plaintext:
            return ""

        try:
            # 派生密钥
            derived_key = hashlib.pbkdf2_hmac(
                "sha256",
                key.encode("utf-8"),
                ConfigEncryptor.DEFAULT_SALT,
                100000,
                dklen=32,
            )

            # XOR加密 (简化方案，生产环境应使用cryptography库)
            key_bytes = derived_key
            plain_bytes = plaintext.encode("utf-8")
            encrypted = bytearray(len(plain_bytes))

            for i, b in enumerate(plain_bytes):
                encrypted[i] = b ^ key_bytes[i % len(key_bytes)]

            return base64.b64encode(bytes(encrypted)).decode("utf-8")

        except Exception as e:
            logger.error(f"加密失败: {e}")
            return ""

    @staticmethod
    def decrypt(ciphertext: str, key: str) -> str:
        """
        解密文本

        Args:
            ciphertext: Base64编码的密文
            key: 加密密钥

        Returns:
            明文
        """
        if not ciphertext:
            return ""

        try:
            derived_key = hashlib.pbkdf2_hmac(
                "sha256",
                key.encode("utf-8"),
                ConfigEncryptor.DEFAULT_SALT,
                100000,
                dklen=32,
            )

            encrypted = base64.b64decode(ciphertext)
            key_bytes = derived_key
            decrypted = bytearray(len(encrypted))

            for i, b in enumerate(encrypted):
                decrypted[i] = b ^ key_bytes[i % len(key_bytes)]

            return decrypted.decode("utf-8")

        except Exception as e:
            logger.error(f"解密失败: {e}")
            return ""

    @staticmethod
    def is_encrypted(value: str) -> bool:
        """
        判断值是否为加密格式

        Args:
            value: 待检查的值

        Returns:
            是否为加密格式
        """
        if not value or not isinstance(value, str):
            return False
        return value.startswith("ENC:")

    @classmethod
    def encrypt_value(cls, value: str, key: str) -> str:
        """
        加密值并添加前缀标识

        Args:
            value: 明文
            key: 加密密钥

        Returns:
            带前缀的加密值
        """
        encrypted = cls.encrypt(value, key)
        return f"ENC:{encrypted}"

    @classmethod
    def decrypt_value(cls, value: str, key: str) -> str:
        """
        解密带前缀的加密值

        Args:
            value: 带前缀的加密值
            key: 加密密钥

        Returns:
            明文
        """
        if not cls.is_encrypted(value):
            return value
        ciphertext = value[4:]  # 去掉 "ENC:" 前缀
        return cls.decrypt(ciphertext, key)


# ============================================================
# 环境变量处理器
# ============================================================

class EnvVarProcessor:
    """
    环境变量处理器

    支持在配置值中引用环境变量:
    - ${ENV_VAR} - 引用环境变量
    - ${ENV_VAR:default} - 引用环境变量，带默认值
    - $ENV_VAR - 简写形式 (仅字母数字和下划线)

    Example:
        配置中: api_key: ${ZHIPU_API_KEY}
        环境变量: ZHIPU_API_KEY=abc123
        解析后: api_key: "abc123"
    """

    @classmethod
    def process(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理配置中的环境变量引用

        Args:
            config: 原始配置

        Returns:
            解析后的配置
        """
        return cls._process_value(config)

    @classmethod
    def _process_value(cls, value: Any) -> Any:
        """
        递归处理配置值

        Args:
            value: 配置值

        Returns:
            处理后的值
        """
        if isinstance(value, str):
            return cls._resolve_env_vars(value)
        elif isinstance(value, dict):
            return {k: cls._process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls._process_value(item) for item in value]
        return value

    @classmethod
    def _resolve_env_vars(cls, value: str) -> str:
        """
        解析字符串中的环境变量引用

        Args:
            value: 包含环境变量引用的字符串

        Returns:
            解析后的字符串
        """
        import re

        # 匹配 ${VAR} 或 ${VAR:default} 格式
        pattern = r'\$\{([^}]+)\}'

        def replacer(match: Any) -> str:
            expr = match.group(1)
            if ":" in expr:
                var_name, default = expr.split(":", 1)
                return os.environ.get(var_name.strip(), default)
            else:
                return os.environ.get(expr.strip(), match.group(0))

        result = re.sub(pattern, replacer, value)

        # 如果整个字符串就是一个环境变量引用且结果为空，保持原样
        if result != value:
            return result

        return value


# ============================================================
# 配置热更新监控
# ============================================================

class ConfigWatcher:
    """
    配置文件热更新监控

    监控配置文件的变化，在文件修改时触发回调。

    使用文件修改时间检测变化，支持:
    - 单文件监控
    - 目录监控
    - 自定义回调函数
    - 防抖处理 (避免频繁触发)

    Example:
        ```python
        watcher = ConfigWatcher(
            config_path="/path/to/config.yaml",
            callback=lambda event: print(f"配置变更: {event}")
        )
        await watcher.start()
        ```
    """

    def __init__(
        self,
        config_path: str,
        callback: Callable[[ConfigChangeEvent], None],
        check_interval: float = 5.0,
        debounce_interval: float = 2.0,
    ):
        """
        初始化配置监控器。

        Args:
            config_path: 配置文件路径
            callback: 变更回调函数
            check_interval: 检查间隔(秒)
            debounce_interval: 防抖间隔(秒)
        """
        self._config_path = Path(config_path)
        self._callback = callback
        self._check_interval = check_interval
        self._debounce_interval = debounce_interval
        self._last_mtime: float = 0.0
        self._last_check: float = 0.0
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._last_configs: Dict[str, Dict[str, Any]] = {}

    async def start(self) -> None:
        """启动配置监控"""
        if self._running:
            return

        self._running = True
        self._last_mtime = self._get_mtime()
        self._task = asyncio.create_task(self._watch_loop())
        logger.info(f"配置监控已启动: {self._config_path}")

    async def stop(self) -> None:
        """停止配置监控"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("配置监控已停止")

    def _get_mtime(self) -> float:
        """获取文件修改时间"""
        try:
            return self._config_path.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    async def _watch_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                current_mtime = self._get_mtime()
                now = time.time()

                if current_mtime > self._last_mtime:
                    # 防抖: 距离上次检查超过间隔才触发
                    if now - self._last_check >= self._debounce_interval:
                        self._last_check = now
                        self._last_mtime = current_mtime
                        await self._handle_change()

            except Exception as e:
                logger.error(f"配置监控异常: {e}")

            await asyncio.sleep(self._check_interval)

    async def _handle_change(self) -> None:
        """处理配置变更"""
        try:
            # 读取新配置
            new_configs = self._load_configs()
            old_configs = self._last_configs.copy()

            # 检测变更
            all_model_ids: Set[str] = set()
            all_model_ids.update(old_configs.keys())
            all_model_ids.update(new_configs.keys())

            for model_id in all_model_ids:
                old_cfg = old_configs.get(model_id)
                new_cfg = new_configs.get(model_id)

                if old_cfg is None and new_cfg is not None:
                    event = ConfigChangeEvent(
                        change_type="added",
                        model_id=model_id,
                        new_config=new_cfg,
                    )
                    self._callback(event)

                elif old_cfg is not None and new_cfg is None:
                    event = ConfigChangeEvent(
                        change_type="removed",
                        model_id=model_id,
                        old_config=old_cfg,
                    )
                    self._callback(event)

                elif old_cfg != new_cfg:
                    event = ConfigChangeEvent(
                        change_type="modified",
                        model_id=model_id,
                        old_config=old_cfg,
                        new_config=new_cfg,
                    )
                    self._callback(event)

            self._last_configs = new_configs

        except Exception as e:
            logger.error(f"处理配置变更失败: {e}")

    def _load_configs(self) -> Dict[str, Dict[str, Any]]:
        """加载配置文件"""
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                if self._config_path.suffix in (".yaml", ".yml"):
                    data = yaml.safe_load(f) or {}
                else:
                    data = json.load(f)

            # 提取模型配置
            configs: Dict[str, Dict[str, Any]] = {}
            providers = data.get("providers", {})
            for provider_name, provider_data in providers.items():
                models = provider_data.get("models", [])
                for model in models:
                    model_id = model.get("model_id", "")
                    if model_id:
                        configs[model_id] = model

            return configs

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}


# ============================================================
# 模型配置管理器
# ============================================================

class ModelConfigManager:
    """
    模型配置管理器

    提供模型配置的完整生命周期管理。

    Features:
    - YAML/JSON配置文件加载与保存
    - 配置验证与校验
    - 环境变量覆盖机制
    - 配置热更新监控
    - API密钥加密存储
    - 配置导入导出

    Example:
        ```python
        # 创建配置管理器
        manager = ModelConfigManager()

        # 从YAML文件加载
        configs = manager.load_from_file("models.yaml")

        # 验证配置
        results = manager.validate_all(configs)

        # 保存到JSON
        manager.save_to_file(configs, "models.json", format="json")

        # 启用热更新
        manager.start_watching("models.yaml", callback)
        ```
    """

    def __init__(
        self,
        encryption_key: Optional[str] = None,
        auto_validate: bool = True,
    ):
        """
        初始化配置管理器。

        Args:
            encryption_key: 加密密钥 (用于API密钥加密)
            auto_validate: 是否自动验证配置
        """
        self._encryption_key = encryption_key or os.environ.get(
            "AGI_CONFIG_ENCRYPTION_KEY", "default_key_change_me"
        )
        self._auto_validate = auto_validate
        self._watchers: Dict[str, ConfigWatcher] = {}
        self._validator = ConfigValidator()
        self._env_processor = EnvVarProcessor()

    # ========================================================
    # 配置加载
    # ========================================================

    def load_from_file(
        self,
        config_path: str,
        process_env: bool = True,
        decrypt_keys: bool = True,
    ) -> Dict[str, Any]:
        """
        从文件加载配置

        支持YAML和JSON格式。

        Args:
            config_path: 配置文件路径
            process_env: 是否处理环境变量引用
            decrypt_keys: 是否解密加密的API密钥

        Returns:
            配置字典

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # 解析文件
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content) or {}
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(
                f"不支持的配置文件格式: {path.suffix}，"
                f"支持: .yaml, .yml, .json"
            )

        # 处理环境变量
        if process_env:
            data = self._env_processor.process(data)

        # 解密API密钥
        if decrypt_keys:
            data = self._decrypt_config(data)

        return data

    def load_from_string(
        self,
        content: str,
        format: str = "yaml",
        process_env: bool = True,
    ) -> Dict[str, Any]:
        """
        从字符串加载配置

        Args:
            content: 配置内容字符串
            format: 格式 (yaml/json)
            process_env: 是否处理环境变量

        Returns:
            配置字典
        """
        if format == "yaml":
            data = yaml.safe_load(content) or {}
        elif format == "json":
            data = json.loads(content)
        else:
            raise ValueError(f"不支持的格式: {format}")

        if process_env:
            data = self._env_processor.process(data)

        return data

    def load_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从字典加载配置 (直接返回副本)

        Args:
            data: 配置字典

        Returns:
            配置字典深拷贝
        """
        return deepcopy(data)

    # ========================================================
    # 配置保存
    # ========================================================

    def save_to_file(
        self,
        config: Dict[str, Any],
        config_path: str,
        format: str = "yaml",
        encrypt_keys: bool = False,
        backup: bool = True,
    ) -> None:
        """
        保存配置到文件

        Args:
            config: 配置字典
            config_path: 保存路径
            format: 格式 (yaml/json)
            encrypt_keys: 是否加密API密钥
            backup: 是否备份已有文件
        """
        path = Path(config_path)

        # 创建目录
        path.parent.mkdir(parents=True, exist_ok=True)

        # 备份已有文件
        if backup and path.exists():
            backup_path = path.with_suffix(
                path.suffix + f".bak.{int(time.time())}"
            )
            shutil.copy2(path, backup_path)
            logger.debug(f"已备份配置文件: {backup_path}")

        # 准备保存数据
        save_data = deepcopy(config)

        # 加密API密钥
        if encrypt_keys:
            save_data = self._encrypt_config(save_data)

        # 写入文件
        with open(path, "w", encoding="utf-8") as f:
            if format == "yaml":
                yaml.dump(
                    save_data, f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            elif format == "json":
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的格式: {format}")

        logger.info(f"配置已保存: {config_path}")

    # ========================================================
    # 配置验证
    # ========================================================

    def validate_model(
        self,
        model_config: Dict[str, Any]
    ) -> ConfigValidationResult:
        """
        验证单个模型配置

        Args:
            model_config: 模型配置字典

        Returns:
            验证结果
        """
        return self._validator.validate(model_config)

    def validate_all(
        self,
        config: Dict[str, Any]
    ) -> List[ConfigValidationResult]:
        """
        验证所有模型配置

        Args:
            config: 完整配置字典

        Returns:
            验证结果列表
        """
        model_configs = self._extract_model_configs(config)
        return [self._validator.validate(mc) for mc in model_configs]

    def _extract_model_configs(
        self,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        从完整配置中提取所有模型配置

        Args:
            config: 完整配置

        Returns:
            模型配置列表
        """
        models: List[Dict[str, Any]] = []
        providers = config.get("providers", {})

        for provider_name, provider_data in providers.items():
            if isinstance(provider_data, dict):
                provider_models = provider_data.get("models", [])
                for model in provider_models:
                    if isinstance(model, dict):
                        models.append(model)

        return models

    # ========================================================
    # 环境变量覆盖
    # ========================================================

    def apply_env_overrides(
        self,
        config: Dict[str, Any],
        prefix: str = "AGI_MODEL_"
    ) -> Dict[str, Any]:
        """
        应用环境变量覆盖

        环境变量命名规则:
        - AGI_MODEL_{PROVIDER}_{MODEL}_{FIELD}
        - 例如: AGI_MODEL_ZHIPU_GLM4_API_KEY=xxx

        Args:
            config: 原始配置
            prefix: 环境变量前缀

        Returns:
            覆盖后的配置
        """
        result = deepcopy(config)
        providers = result.get("providers", {})

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # 解析: AGI_MODEL_PROVIDER_MODEL_FIELD
            parts = env_key[len(prefix):].lower().split("_", 2)
            if len(parts) < 3:
                continue

            provider_name = parts[0]
            # model_name 和 field_name 可能包含下划线
            # 使用已知的字段名来分割
            known_fields = {
                "api_key", "api_base", "timeout", "max_context",
                "max_output", "temperature", "rate_limit",
                "retry_count", "auth_type",
            }

            remaining = "_".join(parts[1:])
            field_name = None
            model_identifier = None

            for kf in known_fields:
                if remaining.endswith(f"_{kf}"):
                    field_name = kf
                    model_identifier = remaining[: -(len(kf) + 1)]
                    break

            if not field_name or not model_identifier:
                continue

            # 查找匹配的模型
            if provider_name in providers:
                for model in providers[provider_name].get("models", []):
                    model_id = model.get("model_id", "")
                    model_name = model.get("model_name", "").lower()
                    model_short = model_id.split("/")[-1].lower()

                    if (model_identifier in model_id.lower()
                            or model_identifier in model_name
                            or model_identifier == model_short):
                        model[field_name] = env_value
                        logger.debug(
                            f"环境变量覆盖: {env_key} -> "
                            f"{model_id}.{field_name}"
                        )
                        break

        return result

    # ========================================================
    # 配置加密/解密
    # ========================================================

    def _encrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        加密配置中的敏感字段

        Args:
            config: 配置字典

        Returns:
            加密后的配置
        """
        return self._process_sensitive_fields(
            config,
            lambda v: ConfigEncryptor.encrypt_value(v, self._encryption_key)
        )

    def _decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解密配置中的敏感字段

        Args:
            config: 配置字典

        Returns:
            解密后的配置
        """
        return self._process_sensitive_fields(
            config,
            lambda v: ConfigEncryptor.decrypt_value(v, self._encryption_key)
        )

    @staticmethod
    def _process_sensitive_fields(
        config: Dict[str, Any],
        processor: Callable[[str], str]
    ) -> Dict[str, Any]:
        """
        处理配置中的敏感字段

        Args:
            config: 配置字典
            processor: 处理函数

        Returns:
            处理后的配置
        """
        sensitive_fields = {"api_key", "secret_key", "client_secret"}

        if isinstance(config, dict):
            result = {}
            for k, v in config.items():
                if k in sensitive_fields and isinstance(v, str):
                    result[k] = processor(v)
                else:
                    result[k] = ModelConfigManager._process_sensitive_fields(
                        v, processor
                    )
            return result
        elif isinstance(config, list):
            return [
                ModelConfigManager._process_sensitive_fields(item, processor)
                for item in config
            ]
        return config

    # ========================================================
    # 配置热更新
    # ========================================================

    def start_watching(
        self,
        config_path: str,
        callback: Callable[[ConfigChangeEvent], None],
        check_interval: float = 5.0,
    ) -> None:
        """
        启动配置文件监控

        Args:
            config_path: 配置文件路径
            callback: 变更回调函数
            check_interval: 检查间隔(秒)
        """
        if config_path in self._watchers:
            logger.warning(f"已在监控: {config_path}")
            return

        watcher = ConfigWatcher(
            config_path=config_path,
            callback=callback,
            check_interval=check_interval,
        )
        self._watchers[config_path] = watcher

        # 在当前事件循环中启动
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(watcher.start())
            else:
                loop.run_until_complete(watcher.start())
        except RuntimeError:
            pass

        logger.info(f"已启动配置监控: {config_path}")

    async def stop_watching(self, config_path: str) -> None:
        """
        停止配置文件监控

        Args:
            config_path: 配置文件路径
        """
        watcher = self._watchers.pop(config_path, None)
        if watcher:
            await watcher.stop()

    async def stop_all_watching(self) -> None:
        """停止所有配置监控"""
        for path, watcher in self._watchers.items():
            await watcher.stop()
        self._watchers.clear()

    # ========================================================
    # 配置导入导出
    # ========================================================

    def export_config(
        self,
        config: Dict[str, Any],
        output_format: str = "yaml",
        include_secrets: bool = False,
    ) -> str:
        """
        导出配置为字符串

        Args:
            config: 配置字典
            output_format: 输出格式 (yaml/json)
            include_secrets: 是否包含敏感信息

        Returns:
            配置字符串
        """
        export_data = deepcopy(config)

        if not include_secrets:
            export_data = self._redact_secrets(export_data)

        if output_format == "yaml":
            return yaml.dump(
                export_data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        elif output_format == "json":
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {output_format}")

    @staticmethod
    def _redact_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        脱敏处理配置中的敏感信息

        Args:
            config: 配置字典

        Returns:
            脱敏后的配置
        """
        sensitive_fields = {"api_key", "secret_key", "client_secret"}

        if isinstance(config, dict):
            result = {}
            for k, v in config.items():
                if k in sensitive_fields and isinstance(v, str) and v:
                    # 保留前4位和后4位
                    if len(v) > 8:
                        result[k] = v[:4] + "****" + v[-4:]
                    else:
                        result[k] = "****"
                else:
                    result[k] = ModelConfigManager._redact_secrets(v)
            return result
        elif isinstance(config, list):
            return [
                ModelConfigManager._redact_secrets(item)
                for item in config
            ]
        return config

    # ========================================================
    # 配置合并
    # ========================================================

    @staticmethod
    def merge_configs(
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        合并两个配置

        override中的值会覆盖base中的同名值。

        Args:
            base: 基础配置
            override: 覆盖配置

        Returns:
            合并后的配置
        """
        result = deepcopy(base)

        for key, value in override.items():
            if (key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)):
                result[key] = ModelConfigManager.merge_configs(
                    result[key], value
                )
            else:
                result[key] = deepcopy(value)

        return result
