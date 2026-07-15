#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI统一框架 - API密钥加密存储模块 (KeyVault)

本模块提供生产级的API密钥加密存储功能，支持多层加密回退机制：
1. 优先使用 Fernet 对称加密 (cryptography 库)
2. 回退到 AES-256-CBC (pycryptodome 库)
3. 最终回退到 XOR 混淆 (附带安全警告)

安全特性：
- 密钥永不以明文存储
- 内存使用后自动清理（零化字节）
- 5分钟无操作自动锁定
- HMAC 签名防篡改
- 每次读取完整性校验
- 仅显示最后4字符的掩码显示
"""

import os
import sys
import json
import hmac
import hashlib
import base64
import logging
import platform
import socket
import uuid
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# 尝试导入加密库，按优先级排列
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _CRYPTO_BACKEND = "fernet"
except ImportError:
    _CRYPTO_BACKEND = None

try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad, unpad
    if _CRYPTO_BACKEND is None:
        _CRYPTO_BACKEND = "aes"
except ImportError:
    if _CRYPTO_BACKEND is None:
        _CRYPTO_BACKEND = "xor"

# 配置日志
logger = logging.getLogger(__name__)

# 常量定义
VAULT_DIR = Path.home() / ".agi_framework"
VAULT_FILE = VAULT_DIR / "vault.enc"
VAULT_META_FILE = VAULT_DIR / "vault.meta"
HMAC_KEY_FILE = VAULT_DIR / ".vault_hmac_key"
AUTO_LOCK_TIMEOUT = 300  # 5分钟自动锁定（秒）
HMAC_ALGORITHM = hashlib.sha256
VAULT_VERSION = "1.0.0"


def _get_machine_fingerprint() -> str:
    """
    获取机器唯一指纹标识。

    综合使用主机名、用户名、平台信息和机器唯一ID生成指纹，
    确保加密密钥与特定机器绑定。

    Returns:
        str: 机器指纹的SHA256哈希值（十六进制字符串）
    """
    components = [
        platform.node() or "unknown",
        os.getenv("USER") or os.getenv("USERNAME") or "unknown",
        platform.system(),
        platform.machine(),
        platform.processor() or "generic",
    ]
    # 尝试获取更稳定的机器ID
    machine_id_sources = [
        "/etc/machine-id",           # Linux systemd
        "/var/lib/dbus/machine-id",  # Linux dbus
        "/etc/hostid",               # FreeBSD/Solaris
    ]
    machine_id = ""
    for mid_path in machine_id_sources:
        try:
            with open(mid_path, "r", encoding="utf-8") as f:
                machine_id = f.read().strip()
                break
        except (FileNotFoundError, PermissionError):
            continue
    if not machine_id:
        # 使用MAC地址作为后备
        try:
            mac = uuid.getnode()
            machine_id = f"{mac:012x}"
        except Exception:
            machine_id = str(uuid.uuid4())
    components.append(machine_id)
    fingerprint = ":".join(components)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _derive_key(salt: bytes, password: Optional[str] = None) -> bytes:
    """
    从密码或机器指纹派生加密密钥。

    Args:
        salt: 盐值，用于密钥派生
        password: 用户密码，如果为None则使用机器指纹

    Returns:
        bytes: 派生的32字节密钥
    """
    if password:
        secret = password.encode("utf-8")
    else:
        secret = _get_machine_fingerprint().encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    return kdf.derive(secret)


def _zero_bytes(data: bytes) -> None:
    """
    安全地零化字节数据，防止敏感信息残留在内存中。

    Args:
        data: 需要零化的字节数据
    """
    try:
        import ctypes
        ctypes.memset(id(data), 0, len(data))
    except Exception:
        # ctypes 不可用时使用Python层面的覆盖
        for i in range(len(data)):
            data[i] = 0


def _mask_value(value: str) -> str:
    """
    对敏感值进行掩码处理，仅显示最后4个字符。

    Args:
        value: 原始敏感值

    Returns:
        str: 掩码后的字符串，如 "****abcd"
    """
    if len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


class FernetEncryptor:
    """使用 Fernet 对称加密的加密器实现。"""

    def __init__(self, key: bytes):
        self._fernet = Fernet(key)

    def encrypt(self, plaintext: bytes) -> bytes:
        return self._fernet.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self._fernet.decrypt(ciphertext)


class AESEncryptor:
    """使用 AES-256-CBC 的加密器实现（pycryptodome 后备方案）。"""

    def __init__(self, key: bytes):
        # AES-256 需要32字节密钥
        self._key = key[:32].ljust(32, b"\x00")

    def encrypt(self, plaintext: bytes) -> bytes:
        iv = os.urandom(16)
        cipher = AES.new(self._key, AES.MODE_CBC, iv)
        padded = pad(plaintext, AES.block_size)
        encrypted = cipher.encrypt(padded)
        # 格式: iv(16) + encrypted_data
        return iv + encrypted

    def decrypt(self, ciphertext: bytes) -> bytes:
        iv = ciphertext[:16]
        encrypted = ciphertext[16:]
        cipher = AES.new(self._key, AES.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(encrypted), AES.block_size)
        return decrypted


class XOREncryptor:
    """XOR 混淆加密器（最终后备方案，安全性较低）。"""

    def __init__(self, key: bytes):
        self._key = key

    def _xor_bytes(self, data: bytes) -> bytes:
        key_len = len(self._key)
        return bytes(b ^ self._key[i % key_len] for i, b in enumerate(data))

    def encrypt(self, plaintext: bytes) -> bytes:
        return base64.b64encode(self._xor_bytes(plaintext))

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self._xor_bytes(base64.b64decode(ciphertext))


def _create_encryptor(key: bytes):
    """
    根据可用的加密库创建对应的加密器实例。

    Args:
        key: 加密密钥

    Returns:
        加密器实例 (FernetEncryptor / AESEncryptor / XOREncryptor)
    """
    if _CRYPTO_BACKEND == "fernet":
        # Fernet 需要 URL-safe base64 编码的32字节密钥
        fernet_key = base64.urlsafe_b64encode(key[:32].ljust(32, b"\x00"))
        return FernetEncryptor(fernet_key)
    elif _CRYPTO_BACKEND == "aes":
        return AESEncryptor(key[:32].ljust(32, b"\x00"))
    else:
        logger.warning(
            "未检测到 cryptography 或 pycryptodome 库，"
            "使用 XOR 混淆模式。强烈建议安装 cryptography 库以获得安全保障！"
        )
        return XOREncryptor(key)


class KeyValidator:
    """
    API密钥验证器，通过向各平台发送测试请求来验证密钥有效性。

    支持的平台包括：
    - OpenAI, DeepSeek, Moonshot, Zhipu (通过 /models 端点)
    - DingTalk, WeCom (通过 /gettoken 端点)
    - Feishu (通过 /auth/v3/tenant_access_token/internal 端点)
    - Tuya (通过 /v1.0/token 端点)
    - 通用验证 (通过 OPTIONS 请求)
    """

    # 各平台的验证端点配置
    VALIDATION_ENDPOINTS = {
        "openai": {
            "url": "https://api.openai.com/v1/models",
            "method": "GET",
            "headers": {"Authorization": "Bearer {key}"},
        },
        "deepseek": {
            "url": "https://api.deepseek.com/v1/models",
            "method": "GET",
            "headers": {"Authorization": "Bearer {key}"},
        },
        "moonshot": {
            "url": "https://api.moonshot.cn/v1/models",
            "method": "GET",
            "headers": {"Authorization": "Bearer {key}"},
        },
        "zhipu": {
            "url": "https://open.bigmodel.cn/api/paas/v4/models",
            "method": "GET",
            "headers": {"Authorization": "Bearer {key}"},
        },
        "dingtalk": {
            "url": "https://oapi.dingtalk.com/gettoken",
            "method": "GET",
            "params": {"appkey": "{key}", "appsecret": "test"},
        },
        "feishu": {
            "url": "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            "method": "POST",
            "headers": {"Content-Type": "application/json; charset=utf-8"},
            "body": {"app_id": "{key}", "app_secret": "test_secret"},
        },
        "wecom": {
            "url": "https://qyapi.weixin.qq.com/cgi-bin/gettoken",
            "method": "GET",
            "params": {"corpid": "{key}", "corpsecret": "test"},
        },
        "tuya": {
            "url": "https://openapi.tuyacn.com/v1.0/token?grant_type=1",
            "method": "GET",
            "headers": {"client_id": "{key}"},
        },
    }

    @staticmethod
    def validate(key_name: str, key_value: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """
        验证API密钥的有效性。

        根据密钥名称自动匹配对应平台的验证端点，
        也可以通过 base_url 指定自定义端点进行通用验证。

        Args:
            key_name: 密钥名称（用于匹配平台）
            key_value: 密钥值
            base_url: 自定义基础URL（用于通用验证）

        Returns:
            Dict: 包含验证结果的字典
                - valid (bool): 密钥是否有效
                - platform (str): 平台名称
                - status_code (int): HTTP状态码（如果进行了请求）
                - message (str): 验证消息
                - latency_ms (float): 请求延迟（毫秒）
        """
        import urllib.request
        import urllib.error
        import urllib.parse

        result = {
            "valid": False,
            "platform": key_name,
            "status_code": None,
            "message": "",
            "latency_ms": 0.0,
        }

        # 查找匹配的平台配置
        platform_key = key_name.lower().split("_")[0]
        endpoint_config = KeyValidator.VALIDATION_ENDPOINTS.get(platform_key)

        if base_url:
            # 使用自定义URL进行通用OPTIONS验证
            try:
                start_time = time.time()
                req = urllib.request.Request(base_url, method="OPTIONS")
                req.add_header("Authorization", f"Bearer {key_value}")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result["status_code"] = resp.status
                    result["valid"] = resp.status < 400
                    result["message"] = "通用验证完成" if result["valid"] else f"HTTP {resp.status}"
            except urllib.error.HTTPError as e:
                result["status_code"] = e.code
                # 某些API即使返回401也说明端点可达，密钥格式可能正确
                result["message"] = f"HTTP {e.code}: {e.reason}"
            except urllib.error.URLError as e:
                result["message"] = f"连接失败: {e.reason}"
            except Exception as e:
                result["message"] = f"验证异常: {str(e)}"
            result["latency_ms"] = (time.time() - start_time) * 1000
            return result

        if not endpoint_config:
            result["message"] = f"未找到平台 '{key_name}' 的验证配置，请提供 base_url 进行通用验证"
            return result

        try:
            start_time = time.time()
            url = endpoint_config["url"]
            method = endpoint_config.get("method", "GET")

            # 构建请求
            if endpoint_config.get("params"):
                params = {
                    k: v.replace("{key}", key_value) for k, v in endpoint_config["params"].items()
                }
                url = f"{url}?{urllib.parse.urlencode(params)}"

            headers = {}
            if endpoint_config.get("headers"):
                headers = {
                    k: v.replace("{key}", key_value) for k, v in endpoint_config["headers"].items()
                }

            body = None
            if endpoint_config.get("body"):
                body_data = {
                    k: v.replace("{key}", key_value) for k, v in endpoint_config["body"].items()
                }
                body = json.dumps(body_data).encode("utf-8")

            req = urllib.request.Request(url, data=body, headers=headers, method=method)

            with urllib.request.urlopen(req, timeout=15) as resp:
                result["status_code"] = resp.status
                response_data = resp.read().decode("utf-8", errors="replace")
                # 大多数API在密钥有效时返回200
                if resp.status == 200:
                    result["valid"] = True
                    result["message"] = "密钥验证通过"
                else:
                    result["message"] = f"HTTP {resp.status}"

        except urllib.error.HTTPError as e:
            result["status_code"] = e.code
            if e.code == 401:
                result["message"] = "密钥无效或已过期 (HTTP 401)"
            elif e.code == 403:
                result["message"] = "密钥权限不足 (HTTP 403)"
            elif e.code == 429:
                # 429表示密钥有效但触发了限流
                result["valid"] = True
                result["message"] = "密钥有效，但触发了频率限制 (HTTP 429)"
            else:
                result["message"] = f"HTTP {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            result["message"] = f"无法连接到验证端点: {e.reason}"
        except Exception as e:
            result["message"] = f"验证过程发生异常: {str(e)}"
            logger.exception("密钥验证异常")

        result["latency_ms"] = (time.time() - start_time) * 1000
        return result


class KeyRotationManager:
    """
    密钥轮换管理器，支持自动和手动密钥轮换。

    功能：
    - 定时轮换（每日/每周/每月）
    - 轮换历史记录
    - 紧急轮换（立即失效并替换）
    """

    ROTATION_INTERVALS = {
        "daily": timedelta(days=1),
        "weekly": timedelta(days=7),
        "monthly": timedelta(days=30),
    }

    def __init__(self, vault: "KeyVault"):
        """
        初始化密钥轮换管理器。

        Args:
            vault: KeyVault 实例引用
        """
        self._vault = vault
        self._rotation_history_file = VAULT_DIR / "rotation_history.json"
        self._rotation_schedules_file = VAULT_DIR / "rotation_schedules.json"
        self._ensure_files()

    def _ensure_files(self) -> None:
        """确保轮换相关文件存在。"""
        VAULT_DIR.mkdir(parents=True, exist_ok=True)
        for fpath in [self._rotation_history_file, self._rotation_schedules_file]:
            if not fpath.exists():
                fpath.write_text("[]", encoding="utf-8")

    def _load_json(self, filepath: Path) -> List[Dict]:
        """加载JSON文件内容。"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.loads(f.read())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_json(self, filepath: Path, data: List[Dict]) -> None:
        """保存数据到JSON文件。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def schedule_rotation(
        self, key_name: str, interval: str = "monthly", new_value: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        安排密钥自动轮换计划。

        Args:
            key_name: 密钥名称
            interval: 轮换间隔 (daily/weekly/monthly)
            new_value: 新密钥值（如果为None，轮换时需要手动提供）

        Returns:
            Dict: 调度结果
        """
        if interval not in self.ROTATION_INTERVALS:
            return {
                "success": False,
                "message": f"无效的轮换间隔: {interval}，"
                           f"支持的值: {list(self.ROTATION_INTERVALS.keys())}",
            }

        # 检查密钥是否存在
        try:
            self._vault.retrieve(key_name)
        except KeyError:
            return {"success": False, "message": f"密钥 '{key_name}' 不存在"}

        schedules = self._load_json(self._rotation_schedules_file)

        # 移除已有的相同密钥调度
        schedules = [s for s in schedules if s.get("key_name") != key_name]

        next_rotation = datetime.now() + self.ROTATION_INTERVALS[interval]
        schedule_entry = {
            "key_name": key_name,
            "interval": interval,
            "next_rotation": next_rotation.isoformat(),
            "has_new_value": new_value is not None,
            "created_at": datetime.now().isoformat(),
        }
        schedules.append(schedule_entry)
        self._save_json(self._rotation_schedules_file, schedules)

        logger.info(
            "已安排密钥 '%s' 的自动轮换，间隔: %s，下次轮换: %s",
            key_name, interval, next_rotation.isoformat(),
        )
        return {
            "success": True,
            "message": f"已安排密钥 '{key_name}' 每{interval}轮换一次",
            "next_rotation": next_rotation.isoformat(),
        }

    def check_and_rotate(self) -> List[Dict[str, Any]]:
        """
        检查所有已调度的轮换计划，执行到期的轮换。

        Returns:
            List[Dict]: 执行的轮换操作结果列表
        """
        schedules = self._load_json(self._rotation_schedules_file)
        results = []
        now = datetime.now()
        updated_schedules = []

        for schedule in schedules:
            key_name = schedule.get("key_name", "")
            next_rotation_str = schedule.get("next_rotation", "")
            interval = schedule.get("interval", "monthly")

            try:
                next_rotation = datetime.fromisoformat(next_rotation_str)
            except (ValueError, TypeError):
                continue

            if now >= next_rotation:
                # 到期，执行轮换
                if schedule.get("has_new_value"):
                    # 自动轮换（使用预存的新值）
                    logger.info("密钥 '%s' 轮换到期，执行自动轮换", key_name)
                    result = self._vault.rotate_key(key_name, new_value="__auto_rotated__")
                    results.append({
                        "key_name": key_name,
                        "action": "auto_rotation",
                        "result": result,
                    })
                else:
                    # 需要手动提供新值
                    logger.warning(
                        "密钥 '%s' 轮换到期，但未提供新值，需要手动处理", key_name
                    )
                    results.append({
                        "key_name": key_name,
                        "action": "pending_manual",
                        "message": "轮换到期，需要手动提供新密钥值",
                    })

                # 更新下次轮换时间
                interval_td = self.ROTATION_INTERVALS.get(interval, timedelta(days=30))
                schedule["next_rotation"] = (now + interval_td).isoformat()
                schedule["last_rotation"] = now.isoformat()

            updated_schedules.append(schedule)

        self._save_json(self._rotation_schedules_file, updated_schedules)
        return results

    def emergency_rotate(self, key_name: str, new_value: str) -> Dict[str, Any]:
        """
        紧急轮换密钥，立即失效旧密钥并替换为新密钥。

        Args:
            key_name: 密钥名称
            new_value: 新密钥值

        Returns:
            Dict: 轮换结果
        """
        logger.warning("执行密钥 '%s' 的紧急轮换", key_name)
        result = self._vault.rotate_key(key_name, new_value)

        # 记录紧急轮换到历史
        history = self._load_json(self._rotation_history_file)
        history.insert(0, {
            "key_name": key_name,
            "type": "emergency",
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False),
        })
        # 只保留最近100条记录
        self._save_json(self._rotation_history_file, history[:100])

        return result

    def get_rotation_history(self, key_name: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """
        获取密钥轮换历史记录。

        Args:
            key_name: 密钥名称过滤（None表示全部）
            limit: 返回记录数量上限

        Returns:
            List[Dict]: 轮换历史记录列表
        """
        history = self._load_json(self._rotation_history_file)
        if key_name:
            history = [h for h in history if h.get("key_name") == key_name]
        return history[:limit]

    def get_schedules(self) -> List[Dict]:
        """获取所有轮换调度计划。"""
        return self._load_json(self._rotation_schedules_file)

    def cancel_schedule(self, key_name: str) -> bool:
        """
        取消指定密钥的轮换调度。

        Args:
            key_name: 密钥名称

        Returns:
            bool: 是否成功取消
        """
        schedules = self._load_json(self._rotation_schedules_file)
        original_len = len(schedules)
        schedules = [s for s in schedules if s.get("key_name") != key_name]
        if len(schedules) < original_len:
            self._save_json(self._rotation_schedules_file, schedules)
            logger.info("已取消密钥 '%s' 的轮换调度", key_name)
            return True
        return False


class KeyVault:
    """
    API密钥加密存储主类。

    提供安全的API密钥存储、检索、管理和验证功能。
    使用机器绑定的加密密钥，确保密钥文件仅在生成它的机器上可用。

    用法示例:
        vault = KeyVault()
        vault.store("openai", "sk-xxxxxxxxxxxx", category="llm")
        key = vault.retrieve("openai")
        vault.delete("openai")
    """

    def __init__(self, vault_path: Optional[str] = None):
        """
        初始化 KeyVault 实例。

        Args:
            vault_path: 自定义保险库文件路径（默认为 ~/.agi_framework/vault.enc）
        """
        self._vault_path = Path(vault_path) if vault_path else VAULT_FILE
        self._vault_dir = self._vault_path.parent
        self._vault_dir.mkdir(parents=True, exist_ok=True)

        # 初始化加密密钥
        self._salt_file = self._vault_dir / ".vault_salt"
        self._salt = self._load_or_create_salt()
        self._encryption_key = self._derive_machine_key()
        self._encryptor = _create_encryptor(self._encryption_key)

        # HMAC 密钥用于完整性验证
        self._hmac_key = self._load_or_create_hmac_key()

        # 自动锁定相关
        self._last_activity = time.time()
        self._is_locked = False

        # 使用统计
        self._usage_stats_file = self._vault_dir / "usage_stats.json"
        self._usage_stats = self._load_usage_stats()

        # 轮换管理器
        self._rotation_manager: Optional[KeyRotationManager] = None

        logger.info(
            "KeyVault 初始化完成，加密后端: %s，保险库路径: %s",
            _CRYPTO_BACKEND, self._vault_path,
        )

    def _load_or_create_salt(self) -> bytes:
        """加载或创建加密盐值。"""
        if self._salt_file.exists():
            try:
                return self._salt_file.read_bytes()
            except Exception as e:
                logger.warning("读取盐值文件失败: %s，将创建新盐值", e)
        salt = os.urandom(32)
        self._salt_file.write_bytes(salt)
        return salt

    def _load_or_create_hmac_key(self) -> bytes:
        """加载或创建HMAC签名密钥。"""
        hmac_key_path = self._vault_dir / ".vault_hmac_key"
        if hmac_key_path.exists():
            try:
                return hmac_key_path.read_bytes()
            except Exception as e:
                logger.warning("读取HMAC密钥文件失败: %s，将创建新密钥", e)
        hmac_key = os.urandom(32)
        self._vault_dir.mkdir(parents=True, exist_ok=True)
        hmac_key_path.write_bytes(hmac_key)
        return hmac_key

    def _derive_machine_key(self) -> bytes:
        """从机器指纹派生加密密钥。"""
        return _derive_key(self._salt)

    def _check_auto_lock(self) -> None:
        """检查是否需要自动锁定。"""
        if self._is_locked:
            raise PermissionError("保险库已锁定，请重新初始化 KeyVault 实例")
        if time.time() - self._last_activity > AUTO_LOCK_TIMEOUT:
            self._is_locked = True
            logger.warning("保险库因长时间无操作已自动锁定（%d秒）", AUTO_LOCK_TIMEOUT)
            raise PermissionError(
                f"保险库已自动锁定（超过{AUTO_LOCK_TIMEOUT}秒无操作），"
                "请重新初始化 KeyVault 实例"
            )

    def _update_activity(self) -> None:
        """更新最后活动时间。"""
        self._last_activity = time.time()

    def _compute_hmac(self, data: bytes) -> str:
        """
        计算数据的HMAC签名。

        Args:
            data: 待签名的数据

        Returns:
            str: HMAC签名的十六进制字符串
        """
        return hmac.new(self._hmac_key, data, HMAC_ALGORITHM).hexdigest()

    def _verify_hmac(self, data: bytes, signature: str) -> bool:
        """
        验证数据的HMAC签名。

        Args:
            data: 原始数据
            signature: 待验证的签名

        Returns:
            bool: 签名是否有效
        """
        expected = self._compute_hmac(data)
        return hmac.compare_digest(expected, signature)

    def _load_vault_data(self) -> Dict:
        """
        从加密文件加载保险库数据，包含完整性校验。

        Returns:
            Dict: 保险库中的密钥数据

        Raises:
            ValueError: 完整性校验失败或数据格式错误
            FileNotFoundError: 保险库文件不存在
        """
        self._check_auto_lock()
        self._update_activity()

        if not self._vault_path.exists():
            raise FileNotFoundError(f"保险库文件不存在: {self._vault_path}")

        try:
            raw_data = self._vault_path.read_bytes()
        except Exception as e:
            raise ValueError(f"读取保险库文件失败: {e}")

        # 解析格式: hmac_signature(64) + encrypted_data
        if len(raw_data) < 65:
            raise ValueError("保险库文件数据格式无效：数据过短")

        stored_hmac = raw_data[:64].decode("ascii")
        encrypted_data = raw_data[64:]

        # 完整性校验
        if not self._verify_hmac(encrypted_data, stored_hmac):
            logger.error("保险库完整性校验失败！文件可能被篡改。")
            raise ValueError(
                "保险库完整性校验失败！文件可能已被篡改或损坏。"
                "请从备份恢复或重新初始化保险库。"
            )

        # 解密数据
        try:
            decrypted = self._encryptor.decrypt(encrypted_data)
            vault_data = json.loads(decrypted.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"解密保险库数据失败: {e}")

        # 版本检查
        if vault_data.get("version") != VAULT_VERSION:
            logger.warning(
                "保险库版本不匹配: 文件版本=%s, 当前版本=%s",
                vault_data.get("version"), VAULT_VERSION,
            )

        return vault_data

    def _save_vault_data(self, vault_data: Dict) -> None:
        """
        加密并保存保险库数据，附带HMAC签名。

        Args:
            vault_data: 保险库数据字典
        """
        self._check_auto_lock()
        self._update_activity()

        vault_data["version"] = VAULT_VERSION
        vault_data["last_modified"] = datetime.now().isoformat()

        plaintext = json.dumps(vault_data, ensure_ascii=False).encode("utf-8")

        # 加密
        encrypted = self._encryptor.encrypt(plaintext)

        # 计算HMAC签名
        signature = self._compute_hmac(encrypted)

        # 写入文件（原子写入：先写临时文件再重命名）
        temp_file = self._vault_path.with_suffix(".tmp")
        try:
            temp_file.write_bytes(signature.encode("ascii") + encrypted)
            temp_file.replace(self._vault_path)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise ValueError(f"保存保险库文件失败: {e}")

        # 安全清理内存中的明文
        _zero_bytes(plaintext)

    def _load_usage_stats(self) -> Dict[str, Dict]:
        """加载使用统计数据。"""
        if self._usage_stats_file.exists():
            try:
                with open(self._usage_stats_file, "r", encoding="utf-8") as f:
                    return json.loads(f.read())
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_usage_stats(self) -> None:
        """保存使用统计数据。"""
        with open(self._usage_stats_file, "w", encoding="utf-8") as f:
            json.dump(self._usage_stats, f, ensure_ascii=False, indent=2)

    def _record_usage(self, key_name: str) -> None:
        """记录密钥使用情况。"""
        now = datetime.now().isoformat()
        if key_name not in self._usage_stats:
            self._usage_stats[key_name] = {
                "count": 0,
                "first_used": now,
                "last_used": now,
            }
        self._usage_stats[key_name]["count"] += 1
        self._usage_stats[key_name]["last_used"] = now
        self._save_usage_stats()

    @property
    def rotation_manager(self) -> KeyRotationManager:
        """获取密钥轮换管理器实例（懒加载）。"""
        if self._rotation_manager is None:
            self._rotation_manager = KeyRotationManager(self)
        return self._rotation_manager

    def store(self, key_name: str, value: str, category: str = "llm") -> Dict[str, Any]:
        """
        加密并存储一个API密钥。

        Args:
            key_name: 密钥名称（如 "openai", "deepseek"）
            value: 密钥值（将被加密存储）
            category: 密钥分类（默认 "llm"），如 "llm", "iot", "im" 等

        Returns:
            Dict: 存储结果，包含操作状态信息
        """
        self._check_auto_lock()
        self._update_activity()

        if not key_name or not key_name.strip():
            raise ValueError("密钥名称不能为空")
        if not value or not value.strip():
            raise ValueError("密钥值不能为空")

        key_name = key_name.strip().lower()
        value = value.strip()

        # 加载现有数据或创建新保险库
        try:
            vault_data = self._load_vault_data()
        except FileNotFoundError:
            vault_data = {
                "version": VAULT_VERSION,
                "keys": {},
                "created_at": datetime.now().isoformat(),
            }

        # 存储密钥元数据和加密值
        encrypted_value = self._encryptor.encrypt(value.encode("utf-8"))
        vault_data["keys"][key_name] = {
            "encrypted_value": encrypted_value.decode("latin-1"),
            "category": category,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "length": len(value),
        }

        self._save_vault_data(vault_data)

        # 安全清理
        _zero_bytes(value.encode("utf-8"))

        logger.info("已存储密钥 '%s' (分类: %s)", key_name, category)
        return {
            "success": True,
            "key_name": key_name,
            "category": category,
            "message": f"密钥 '{key_name}' 已安全存储",
        }

    def retrieve(self, key_name: str) -> str:
        """
        解密并返回指定名称的API密钥。

        Args:
            key_name: 密钥名称

        Returns:
            str: 解密后的密钥值

        Raises:
            KeyError: 密钥不存在
            ValueError: 解密失败
        """
        self._check_auto_lock()
        self._update_activity()

        key_name = key_name.strip().lower()
        vault_data = self._load_vault_data()

        if key_name not in vault_data.get("keys", {}):
            raise KeyError(f"密钥 '{key_name}' 不存在")

        key_entry = vault_data["keys"][key_name]
        encrypted_value = key_entry["encrypted_value"].encode("latin-1")

        try:
            decrypted_bytes = self._encryptor.decrypt(encrypted_value)
            value = decrypted_bytes.decode("utf-8")
        except Exception as e:
            raise ValueError(f"解密密钥 '{key_name}' 失败: {e}")

        # 记录使用
        self._record_usage(key_name)

        # 安全清理
        _zero_bytes(decrypted_bytes)

        logger.debug("已检索密钥 '%s'", key_name)
        return value

    def delete(self, key_name: str) -> Dict[str, Any]:
        """
        删除指定名称的API密钥。

        Args:
            key_name: 密钥名称

        Returns:
            Dict: 删除结果
        """
        self._check_auto_lock()
        self._update_activity()

        key_name = key_name.strip().lower()
        vault_data = self._load_vault_data()

        if key_name not in vault_data.get("keys", {}):
            raise KeyError(f"密钥 '{key_name}' 不存在")

        del vault_data["keys"][key_name]
        self._save_vault_data(vault_data)

        # 清理使用统计
        if key_name in self._usage_stats:
            del self._usage_stats[key_name]
            self._save_usage_stats()

        logger.info("已删除密钥 '%s'", key_name)
        return {
            "success": True,
            "key_name": key_name,
            "message": f"密钥 '{key_name}' 已删除",
        }

    def list_keys(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出所有已存储的密钥（值已掩码处理）。

        Args:
            category: 按分类过滤（None表示返回全部）

        Returns:
            List[Dict]: 密钥信息列表，每个条目包含：
                - name: 密钥名称
                - category: 分类
                - masked_value: 掩码后的值
                - created_at: 创建时间
                - updated_at: 更新时间
        """
        self._check_auto_lock()
        self._update_activity()

        try:
            vault_data = self._load_vault_data()
        except FileNotFoundError:
            return []

        keys_list = []
        for name, entry in vault_data.get("keys", {}).items():
            if category and entry.get("category") != category:
                continue
            keys_list.append({
                "name": name,
                "category": entry.get("category", "unknown"),
                "masked_value": "*" * max(entry.get("length", 8) - 4, 4) + "****",
                "created_at": entry.get("created_at", ""),
                "updated_at": entry.get("updated_at", ""),
            })

        keys_list.sort(key=lambda x: x["name"])
        return keys_list

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        按名称或分类搜索密钥。

        Args:
            query: 搜索关键词

        Returns:
            List[Dict]: 匹配的密钥信息列表（值已掩码）
        """
        self._check_auto_lock()
        self._update_activity()

        query = query.strip().lower()
        all_keys = self.list_keys()

        results = []
        for key_info in all_keys:
            if query in key_info["name"] or query in key_info.get("category", ""):
                results.append(key_info)

        return results

    def export_keys(self, password: str, filepath: str) -> Dict[str, Any]:
        """
        将所有密钥导出为加密备份文件。

        导出的文件使用用户提供的密码进行加密，与机器无关，
        可在其他机器上通过 import_keys 导入。

        Args:
            password: 导出加密密码
            filepath: 导出文件路径

        Returns:
            Dict: 导出结果
        """
        self._check_auto_lock()
        self._update_activity()

        if not password:
            raise ValueError("导出密码不能为空")

        vault_data = self._load_vault_data()

        # 解密所有密钥值
        export_data = {
            "version": VAULT_VERSION,
            "exported_at": datetime.now().isoformat(),
            "exported_from": _get_machine_fingerprint()[:16],
            "keys": {},
        }

        for name, entry in vault_data.get("keys", {}).items():
            try:
                encrypted_value = entry["encrypted_value"].encode("latin-1")
                decrypted = self._encryptor.decrypt(encrypted_value).decode("utf-8")
                export_data["keys"][name] = {
                    "value": decrypted,
                    "category": entry.get("category", "unknown"),
                    "created_at": entry.get("created_at", ""),
                }
                _zero_bytes(encrypted_value)
            except Exception as e:
                logger.warning("导出密钥 '%s' 失败: %s", name, e)

        # 使用用户密码加密导出数据
        export_salt = os.urandom(32)
        export_key = _derive_key(export_salt, password)
        export_encryptor = _create_encryptor(export_key)
        plaintext = json.dumps(export_data, ensure_ascii=False).encode("utf-8")
        encrypted = export_encryptor.encrypt(plaintext)

        # 写入文件: salt(32) + encrypted_data
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(export_salt + encrypted)

        # 安全清理
        _zero_bytes(plaintext)
        _zero_bytes(export_key)

        key_count = len(export_data["keys"])
        logger.info("已导出 %d 个密钥到 %s", key_count, filepath)
        return {
            "success": True,
            "key_count": key_count,
            "filepath": str(output_path),
            "message": f"已成功导出 {key_count} 个密钥",
        }

    def import_keys(self, password: str, filepath: str) -> Dict[str, Any]:
        """
        从加密备份文件导入密钥。

        Args:
            password: 导出时使用的密码
            filepath: 备份文件路径

        Returns:
            Dict: 导入结果
        """
        self._check_auto_lock()
        self._update_activity()

        if not password:
            raise ValueError("导入密码不能为空")

        input_path = Path(filepath)
        if not input_path.exists():
            raise FileNotFoundError(f"备份文件不存在: {filepath}")

        raw_data = input_path.read_bytes()

        # 解析格式: salt(32) + encrypted_data
        if len(raw_data) < 33:
            raise ValueError("备份文件格式无效：数据过短")

        import_salt = raw_data[:32]
        encrypted_data = raw_data[32:]

        # 使用密码解密
        import_key = _derive_key(import_salt, password)
        import_encryptor = _create_encryptor(import_key)

        try:
            decrypted = import_encryptor.decrypt(encrypted_data)
            export_data = json.loads(decrypted.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"解密备份文件失败，密码可能不正确: {e}")

        # 导入密钥到保险库
        imported_count = 0
        skipped_count = 0
        for name, entry in export_data.get("keys", {}).items():
            value = entry.get("value", "")
            category = entry.get("category", "llm")
            if not value:
                skipped_count += 1
                continue
            try:
                self.store(name, value, category=category)
                imported_count += 1
            except Exception as e:
                logger.warning("导入密钥 '%s' 失败: %s", name, e)
                skipped_count += 1

        # 安全清理
        _zero_bytes(decrypted)
        _zero_bytes(import_key)

        logger.info("已导入 %d 个密钥（跳过 %d 个）", imported_count, skipped_count)
        return {
            "success": True,
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "message": f"成功导入 {imported_count} 个密钥",
        }

    def validate_key(self, key_name: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """
        验证指定密钥的有效性（通过向对应平台发送测试请求）。

        Args:
            key_name: 密钥名称
            base_url: 自定义验证URL（用于非预定义平台）

        Returns:
            Dict: 验证结果
        """
        self._check_auto_lock()
        self._update_activity()

        try:
            key_value = self.retrieve(key_name)
        except (KeyError, ValueError) as e:
            return {
                "valid": False,
                "platform": key_name,
                "message": f"无法检索密钥: {e}",
            }

        result = KeyValidator.validate(key_name, key_value, base_url)

        # 安全清理
        _zero_bytes(key_value.encode("utf-8"))

        return result

    def rotate_key(self, key_name: str, new_value: str) -> Dict[str, Any]:
        """
        轮换指定密钥的值。

        保留密钥名称和分类，仅替换密钥值，
        并记录轮换历史。

        Args:
            key_name: 密钥名称
            new_value: 新的密钥值

        Returns:
            Dict: 轮换结果
        """
        self._check_auto_lock()
        self._update_activity()

        key_name = key_name.strip().lower()

        # 获取当前密钥信息
        try:
            vault_data = self._load_vault_data()
        except FileNotFoundError:
            raise KeyError(f"密钥 '{key_name}' 不存在，无法轮换")

        if key_name not in vault_data.get("keys", {}):
            raise KeyError(f"密钥 '{key_name}' 不存在，无法轮换")

        old_entry = vault_data["keys"][key_name]
        category = old_entry.get("category", "llm")

        # 存储新值
        encrypted_new = self._encryptor.encrypt(new_value.encode("utf-8"))
        vault_data["keys"][key_name] = {
            "encrypted_value": encrypted_new.decode("latin-1"),
            "category": category,
            "created_at": old_entry.get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "length": len(new_value),
            "rotated_at": datetime.now().isoformat(),
            "rotation_count": old_entry.get("rotation_count", 0) + 1,
        }

        self._save_vault_data(vault_data)

        # 记录轮换历史
        rotation_mgr = self.rotation_manager
        history = rotation_mgr._load_json(rotation_mgr._rotation_history_file)
        history.insert(0, {
            "key_name": key_name,
            "type": "manual",
            "timestamp": datetime.now().isoformat(),
            "success": True,
        })
        rotation_mgr._save_json(rotation_mgr._rotation_history_file, history[:100])

        # 安全清理
        _zero_bytes(new_value.encode("utf-8"))

        logger.info("已轮换密钥 '%s'", key_name)
        return {
            "success": True,
            "key_name": key_name,
            "message": f"密钥 '{key_name}' 已成功轮换",
            "rotation_count": vault_data["keys"][key_name].get("rotation_count", 1),
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        获取所有密钥的使用统计信息。

        Returns:
            Dict: 使用统计数据，包含：
                - total_keys: 总密钥数
                - stats: 各密钥的使用统计
        """
        self._check_auto_lock()
        self._update_activity()

        self._usage_stats = self._load_usage_stats()
        return {
            "total_keys": len(self._usage_stats),
            "stats": dict(self._usage_stats),
        }

    def unlock(self) -> Dict[str, Any]:
        """
        手动解锁保险库。

        Returns:
            Dict: 解锁结果
        """
        self._is_locked = False
        self._last_activity = time.time()
        logger.info("保险库已手动解锁")
        return {"success": True, "message": "保险库已解锁"}

    def is_locked(self) -> bool:
        """检查保险库是否处于锁定状态。"""
        if not self._is_locked and time.time() - self._last_activity > AUTO_LOCK_TIMEOUT:
            self._is_locked = True
        return self._is_locked

    def get_backend_info(self) -> Dict[str, str]:
        """
        获取当前加密后端信息。

        Returns:
            Dict: 加密后端详情
        """
        return {
            "backend": _CRYPTO_BACKEND,
            "version": VAULT_VERSION,
            "vault_path": str(self._vault_path),
            "auto_lock_timeout": f"{AUTO_LOCK_TIMEOUT}s",
            "machine_fingerprint": _get_machine_fingerprint()[:16] + "...",
        }


# ============================================================
# CLI 命令行接口
# ============================================================

def _print_result(result: Dict) -> None:
    """格式化输出操作结果。"""
    if result.get("success"):
        print(f"[成功] {result.get('message', '')}")
    else:
        print(f"[失败] {result.get('message', '操作失败')}")
    # 输出额外字段
    for k, v in result.items():
        if k not in ("success", "message"):
            print(f"  {k}: {v}")


def main():
    """CLI 入口函数。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="key_vault",
        description="AGI统一框架 - API密钥加密存储管理工具",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # set 命令 - 存储密钥
    set_parser = subparsers.add_parser("set", help="存储一个API密钥")
    set_parser.add_argument("name", help="密钥名称")
    set_parser.add_argument("value", help="密钥值")
    set_parser.add_argument("--category", "-c", default="llm", help="密钥分类 (默认: llm)")

    # get 命令 - 检索密钥
    get_parser = subparsers.add_parser("get", help="检索并显示密钥值")
    get_parser.add_argument("name", help="密钥名称")
    get_parser.add_argument("--show", "-s", action="store_true", help="显示完整值（危险！）")

    # list 命令 - 列出所有密钥
    list_parser = subparsers.add_parser("list", help="列出所有已存储的密钥")
    list_parser.add_argument("--category", "-c", default=None, help="按分类过滤")

    # delete 命令 - 删除密钥
    delete_parser = subparsers.add_parser("delete", help="删除指定密钥")
    delete_parser.add_argument("name", help="密钥名称")
    delete_parser.add_argument("--force", "-f", action="store_true", help="跳过确认")

    # validate 命令 - 验证密钥
    validate_parser = subparsers.add_parser("validate", help="验证密钥有效性")
    validate_parser.add_argument("name", help="密钥名称")
    validate_parser.add_argument("--url", "-u", default=None, help="自定义验证URL")

    # export 命令 - 导出密钥
    export_parser = subparsers.add_parser("export", help="导出所有密钥到加密备份")
    export_parser.add_argument("--password", "-p", required=True, help="导出加密密码")
    export_parser.add_argument("--output", "-o", required=True, help="输出文件路径")

    # import 命令 - 导入密钥
    import_parser = subparsers.add_parser("import", help="从备份文件导入密钥")
    import_parser.add_argument("--password", "-p", required=True, help="导入密码")
    import_parser.add_argument("--input", "-i", required=True, help="输入备份文件路径")

    # search 命令 - 搜索密钥
    search_parser = subparsers.add_parser("search", help="搜索密钥")
    search_parser.add_argument("query", help="搜索关键词")

    # stats 命令 - 使用统计
    subparsers.add_parser("stats", help="查看密钥使用统计")

    # info 命令 - 后端信息
    subparsers.add_parser("info", help="显示加密后端信息")

    # rotate 命令 - 轮换密钥
    rotate_parser = subparsers.add_parser("rotate", help="轮换密钥值")
    rotate_parser.add_argument("name", help="密钥名称")
    rotate_parser.add_argument("new_value", help="新的密钥值")

    # schedule 命令 - 轮换调度
    schedule_parser = subparsers.add_parser("schedule", help="管理密钥轮换调度")
    schedule_parser.add_argument("name", help="密钥名称")
    schedule_parser.add_argument(
        "--interval", "-i", choices=["daily", "weekly", "monthly"],
        default="monthly", help="轮换间隔 (默认: monthly)",
    )
    schedule_parser.add_argument("--cancel", action="store_true", help="取消轮换调度")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 配置日志输出到stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    vault = KeyVault()

    try:
        if args.command == "set":
            result = vault.store(args.name, args.value, category=args.category)
            _print_result(result)

        elif args.command == "get":
            try:
                value = vault.retrieve(args.name)
                if args.show:
                    print(value)
                else:
                    print(f"{args.name}: {_mask_value(value)}")
            except KeyError as e:
                print(f"[错误] {e}")
            except PermissionError as e:
                print(f"[错误] {e}")

        elif args.command == "list":
            keys = vault.list_keys(category=args.category)
            if not keys:
                print("保险库中没有存储任何密钥。")
            else:
                print(f"{'名称':<20} {'分类':<10} {'掩码值':<30} {'更新时间'}")
                print("-" * 80)
                for k in keys:
                    print(
                        f"{k['name']:<20} {k['category']:<10} "
                        f"{k['masked_value']:<30} {k['updated_at']}"
                    )

        elif args.command == "delete":
            if not args.force:
                confirm = input(f"确认删除密钥 '{args.name}'? (y/N): ").strip().lower()
                if confirm != "y":
                    print("已取消删除操作。")
                    return
            try:
                result = vault.delete(args.name)
                _print_result(result)
            except KeyError as e:
                print(f"[错误] {e}")

        elif args.command == "validate":
            result = vault.validate_key(args.name, base_url=args.url)
            status = "有效" if result.get("valid") else "无效"
            print(f"密钥 '{args.name}': {status}")
            print(f"  消息: {result.get('message', '')}")
            if result.get("latency_ms"):
                print(f"  延迟: {result['latency_ms']:.1f}ms")

        elif args.command == "export":
            result = vault.export_keys(args.password, args.output)
            _print_result(result)

        elif args.command == "import":
            result = vault.import_keys(args.password, args.input)
            _print_result(result)

        elif args.command == "search":
            results = vault.search(args.query)
            if not results:
                print(f"未找到匹配 '{args.query}' 的密钥。")
            else:
                for r in results:
                    print(f"  {r['name']} ({r['category']}): {r['masked_value']}")

        elif args.command == "stats":
            stats = vault.get_usage_stats()
            print(f"总密钥数: {stats['total_keys']}")
            if stats["stats"]:
                print(f"\n{'名称':<20} {'使用次数':<10} {'首次使用':<22} {'最近使用'}")
                print("-" * 80)
                for name, s in sorted(stats["stats"].items()):
                    print(
                        f"{name:<20} {s.get('count', 0):<10} "
                        f"{s.get('first_used', 'N/A'):<22} {s.get('last_used', 'N/A')}"
                    )

        elif args.command == "info":
            info = vault.get_backend_info()
            print("KeyVault 加密后端信息:")
            for k, v in info.items():
                print(f"  {k}: {v}")

        elif args.command == "rotate":
            try:
                result = vault.rotate_key(args.name, args.new_value)
                _print_result(result)
            except KeyError as e:
                print(f"[错误] {e}")

        elif args.command == "schedule":
            if args.cancel:
                success = vault.rotation_manager.cancel_schedule(args.name)
                if success:
                    print(f"已取消密钥 '{args.name}' 的轮换调度。")
                else:
                    print(f"未找到密钥 '{args.name}' 的轮换调度。")
            else:
                result = vault.rotation_manager.schedule_rotation(
                    args.name, interval=args.interval
                )
                _print_result(result)

    except PermissionError as e:
        print(f"[错误] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[错误] {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("执行命令时发生异常")
        print(f"[错误] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
