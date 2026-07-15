"""
插件安装器模块

提供插件包下载、依赖解析、执行pip安装、签名验证和失败回滚功能。
仅使用 Python 标准库。
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------

@dataclass
class InstallResult:
    """安装结果"""
    success: bool
    plugin_name: str
    message: str = ""
    installed_files: List[str] = field(default_factory=list)
    installed_packages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration: float = 0.0
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class PackageInfo:
    """包信息"""
    name: str
    version: str
    url: str
    checksum: str = ""
    signature_url: str = ""
    dependencies: List[str] = field(default_factory=list)
    size: int = 0


@dataclass
class RollbackAction:
    """回滚操作"""
    action_type: str  # "file_delete", "file_restore", "pip_uninstall", "dir_remove"
    target: str
    backup_path: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ---------------------------------------------------------------------------
# 签名验证器
# ---------------------------------------------------------------------------

class SignatureVerifier:
    """签名验证器

    验证插件包的数字签名，确保包来源可信且未被篡改。
    """

    def __init__(self, trusted_keys: Optional[Dict[str, str]] = None):
        """
        Args:
            trusted_keys: 可信的公钥字典 {key_id: public_key}
        """
        self._trusted_keys = trusted_keys or {}
        self._verification_cache: Dict[str, bool] = {}

    def add_trusted_key(self, key_id: str, public_key: str) -> None:
        """添加可信公钥"""
        self._trusted_keys[key_id] = public_key

    def remove_trusted_key(self, key_id: str) -> bool:
        """移除可信公钥"""
        if key_id in self._trusted_keys:
            del self._trusted_keys[key_id]
            return True
        return False

    def verify_checksum(self, file_path: str, expected_checksum: str,
                        algorithm: str = "sha256") -> bool:
        """验证文件校验和

        Args:
            file_path: 文件路径
            expected_checksum: 期望的校验和
            algorithm: 哈希算法 (md5, sha1, sha256, sha512)

        Returns:
            校验是否通过
        """
        if not os.path.exists(file_path):
            return False

        hash_func = self._get_hash_function(algorithm)
        if hash_func is None:
            return False

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)

            actual_checksum = hash_func.hexdigest()
            return actual_checksum.lower() == expected_checksum.lower()
        except (IOError, OSError):
            return False

    def verify_signature(self, file_path: str, signature: str,
                        public_key: str) -> bool:
        """验证数字签名

        使用GPG或PGP验证签名。

        Args:
            file_path: 文件路径
            signature: 签名数据
            public_key: 公钥

        Returns:
            签名是否有效
        """
        cache_key = f"{file_path}:{signature[:16]}"
        if cache_key in self._verification_cache:
            return self._verification_cache[cache_key]

        # 尝试使用gpg验证
        result = self._verify_with_gpg(file_path, signature, public_key)
        self._verification_cache[cache_key] = result
        return result

    def _verify_with_gpg(self, file_path: str, signature: str,
                        public_key: str) -> bool:
        """使用GPG验证签名"""
        try:
            # 创建临时目录存储密钥和签名
            with tempfile.TemporaryDirectory() as tmpdir:
                key_file = os.path.join(tmpdir, "pubkey.gpg")
                sig_file = os.path.join(tmpdir, "signature.sig")

                # 写入公钥
                with open(key_file, "w") as f:
                    f.write(public_key)

                # 写入签名
                with open(sig_file, "w") as f:
                    f.write(signature)

                # 执行gpg验证
                cmd = [
                    "gpg", "--batch", "--verify",
                    "--keyring", key_file,
                    sig_file, file_path
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                return result.returncode == 0

        except (subprocess.TimeoutExpired, FileNotFoundError,
                subprocess.SubprocessError):
            return False

    def verify_package(self, package_path: str, package_info: PackageInfo) -> Tuple[bool, str]:
        """验证完整的包

        Args:
            package_path: 包文件路径
            package_info: 包信息

        Returns:
            (是否验证通过, 错误信息)
        """
        # 验证校验和
        if package_info.checksum:
            if not self.verify_checksum(package_path, package_info.checksum):
                return False, f"校验和验证失败: {package_info.checksum}"

        # 验证签名（如果提供）
        if package_info.signature_url:
            try:
                # 下载签名
                signature = self._download_content(package_info.signature_url)
                if not signature:
                    return False, "无法下载签名文件"

                # 获取可信公钥（简化处理）
                public_key = self._get_default_public_key()
                if not public_key:
                    return False, "无可用的公钥进行签名验证"

                if not self.verify_signature(package_path, signature, public_key):
                    return False, "签名验证失败"

            except Exception as e:
                return False, f"签名验证出错: {str(e)}"

        return True, ""

    def _get_hash_function(self, algorithm: str) -> Optional[Any]:
        """获取哈希函数"""
        algorithms = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
        }
        return algorithms.get(algorithm.lower())

    def _download_content(self, url: str) -> Optional[str]:
        """下载内容"""
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read().decode("utf-8")
        except Exception:
            return None

    def _get_default_public_key(self) -> str:
        """获取默认公钥"""
        return self._trusted_keys.get("default", "")


# ---------------------------------------------------------------------------
# 回滚管理器
# ---------------------------------------------------------------------------

class RollbackManager:
    """回滚管理器

    记录安装过程中的操作，支持在失败时回滚到之前的状态。
    """

    def __init__(self, backup_dir: Optional[str] = None):
        """
        Args:
            backup_dir: 备份目录路径
        """
        self._backup_dir = backup_dir or os.path.join(
            tempfile.gettempdir(), "plugin_rollback"
        )
        self._actions: deque = deque()
        self._enabled = True
        self._session_id = f"rollback_{int(time.time() * 1000)}"
        self._session_dir = os.path.join(self._backup_dir, self._session_id)

        os.makedirs(self._session_dir, exist_ok=True)

    def enable(self) -> None:
        """启用回滚"""
        self._enabled = True

    def disable(self) -> None:
        """禁用回滚"""
        self._enabled = False

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self._enabled

    def record_file_backup(self, file_path: str) -> Optional[str]:
        """记录文件备份

        Args:
            file_path: 文件路径

        Returns:
            备份文件路径，失败返回None
        """
        if not self._enabled:
            return None

        if not os.path.exists(file_path):
            return None

        try:
            backup_name = hashlib.md5(file_path.encode()).hexdigest()
            backup_path = os.path.join(self._session_dir, backup_name)

            # 复制文件到备份目录
            if os.path.isdir(file_path):
                backup_path += "_dir"
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                shutil.copytree(file_path, backup_path)
            else:
                shutil.copy2(file_path, backup_path)

            self._actions.append(RollbackAction(
                action_type="file_restore",
                target=file_path,
                backup_path=backup_path,
            ))

            return backup_path

        except (IOError, OSError, shutil.Error):
            return None

    def record_file_delete(self, file_path: str) -> None:
        """记录文件删除操作"""
        if not self._enabled:
            return

        self._actions.append(RollbackAction(
            action_type="file_delete",
            target=file_path,
        ))

    def record_directory_remove(self, dir_path: str) -> None:
        """记录目录删除操作"""
        if not self._enabled:
            return

        self._actions.append(RollbackAction(
            action_type="dir_remove",
            target=dir_path,
        ))

    def record_pip_install(self, package_name: str, version: str = "") -> None:
        """记录pip安装操作"""
        if not self._enabled:
            return

        self._actions.append(RollbackAction(
            action_type="pip_uninstall",
            target=package_name,
            backup_path=version,
        ))

    def rollback(self) -> Tuple[bool, List[str]]:
        """执行回滚

        Returns:
            (回滚是否成功, 错误信息列表)
        """
        if not self._enabled:
            return True, []

        errors: List[str] = []

        # 逆序执行回滚
        while self._actions:
            action = self._actions.pop()
            success = self._execute_action(action)
            if not success:
                errors.append(f"回滚失败: {action.action_type} - {action.target}")

        # 清理备份目录
        self._cleanup_session_dir()

        return len(errors) == 0, errors

    def _execute_action(self, action: RollbackAction) -> bool:
        """执行单个回滚操作"""
        try:
            if action.action_type == "file_restore":
                if os.path.exists(action.backup_path):
                    if os.path.isdir(action.backup_path):
                        if os.path.exists(action.target):
                            shutil.rmtree(action.target)
                        shutil.copytree(action.backup_path, action.target)
                    else:
                        shutil.copy2(action.backup_path, action.target)
                    return True
                return False

            elif action.action_type == "file_delete":
                if os.path.exists(action.target):
                    if os.path.isfile(action.target):
                        os.remove(action.target)
                    return True
                return True  # 文件不存在也算成功

            elif action.action_type == "dir_remove":
                if os.path.exists(action.target):
                    shutil.rmtree(action.target)
                return True

            elif action.action_type == "pip_uninstall":
                return self._pip_uninstall(action.target)

            return False

        except (IOError, OSError, subprocess.SubprocessError):
            return False

    def _pip_uninstall(self, package_name: str) -> bool:
        """卸载pip包"""
        try:
            result = subprocess.run(
                [self._get_python_path(), "-m", "pip", "uninstall",
                 "-y", package_name],
                capture_output=True,
                timeout=120
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False

    def _get_python_path(self) -> str:
        """获取Python解释器路径"""
        return os.environ.get("PYTHON_PATH", "python")

    def _cleanup_session_dir(self) -> None:
        """清理会话目录"""
        try:
            if os.path.exists(self._session_dir):
                shutil.rmtree(self._session_dir)
        except (IOError, OSError):
            pass

    def clear(self) -> None:
        """清除所有回滚记录"""
        self._actions.clear()
        self._cleanup_session_dir()


# ---------------------------------------------------------------------------
# 包下载器
# ---------------------------------------------------------------------------

class PackageDownloader:
    """包下载器

    从URL下载插件包，支持重试、进度跟踪和校验和验证。
    """

    def __init__(self, timeout: int = 300, max_retries: int = 3,
                 chunk_size: int = 8192):
        """
        Args:
            timeout: 下载超时（秒）
            max_retries: 最大重试次数
            chunk_size: 每次读取的块大小
        """
        self._timeout = timeout
        self._max_retries = max_retries
        self._chunk_size = chunk_size
        self._download_cache: Dict[str, str] = {}

    def download(self, url: str, dest_path: str,
                progress_callback: Optional[Callable[[int, int], None]] = None,
                expected_checksum: Optional[str] = None,
                algorithm: str = "sha256") -> Tuple[bool, str]:
        """下载包

        Args:
            url: 下载URL
            dest_path: 目标路径
            progress_callback: 进度回调 (已下载字节数, 总字节数)
            expected_checksum: 期望的校验和

        Returns:
            (是否成功, 错误信息)
        """
        # 检查缓存
        if url in self._download_cache:
            cached_path = self._download_cache[url]
            if os.path.exists(cached_path):
                try:
                    shutil.copy2(cached_path, dest_path)
                    return True, ""
                except (IOError, OSError):
                    pass

        # 创建目标目录
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        # 执行下载
        for attempt in range(self._max_retries):
            try:
                success, error = self._download_with_progress(
                    url, dest_path, progress_callback
                )
                if not success:
                    continue

                # 验证校验和
                if expected_checksum:
                    verifier = SignatureVerifier()
                    if not verifier.verify_checksum(dest_path, expected_checksum, algorithm):
                        os.remove(dest_path)
                        return False, f"校验和验证失败"

                # 更新缓存
                self._download_cache[url] = dest_path
                return True, ""

            except Exception as e:
                if attempt == self._max_retries - 1:
                    return False, f"下载失败: {str(e)}"

        return False, f"达到最大重试次数 ({self._max_retries})"

    def _download_with_progress(self, url: str, dest_path: str,
                               progress_callback: Optional[Callable[[int, int], None]]) -> Tuple[bool, str]:
        """带进度的下载"""
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0

                with open(dest_path, "wb") as f:
                    while True:
                        chunk = response.read(self._chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)

                return True, ""

        except urllib.error.HTTPError as e:
            return False, f"HTTP错误: {e.code} {e.reason}"
        except urllib.error.URLError as e:
            return False, f"URL错误: {str(e.reason)}"
        except TimeoutError:
            return False, "下载超时"

    def extract_package(self, archive_path: str, dest_dir: str) -> Tuple[bool, str, List[str]]:
        """解压包

        Args:
            archive_path: 压缩包路径
            dest_dir: 目标目录

        Returns:
            (是否成功, 错误信息, 提取的文件列表)
        """
        os.makedirs(dest_dir, exist_ok=True)
        extracted_files: List[str] = []

        try:
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    extracted_files = zf.namelist()
                    zf.extractall(dest_dir)
            elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
                import tarfile
                with tarfile.open(archive_path, "r:gz") as tf:
                    extracted_files = tf.getnames()
                    tf.extractall(dest_dir)
            else:
                return False, f"不支持的压缩格式: {archive_path}", []

            return True, "", extracted_files

        except (zipfile.BadZipFile, tarfile.TarError, IOError, OSError) as e:
            return False, f"解压失败: {str(e)}", []

    def clear_cache(self) -> None:
        """清除下载缓存"""
        self._download_cache.clear()


# ---------------------------------------------------------------------------
# 依赖解析器（扩展版）
# ---------------------------------------------------------------------------

class InstallerDependencyResolver:
    """安装器依赖解析器

    解析插件依赖，确定安装顺序，检测冲突。
    """

    def __init__(self):
        self._installed_packages: Dict[str, str] = {}
        self._plugin_dependencies: Dict[str, List[str]] = {}

    def set_installed_packages(self, packages: Dict[str, str]) -> None:
        """设置已安装的包"""
        self._installed_packages = packages.copy()

    def get_installed_version(self, package_name: str) -> Optional[str]:
        """获取已安装包的版本"""
        return self._installed_packages.get(package_name)

    def resolve_dependencies(self, plugin_name: str,
                             dependencies: List[Tuple[str, str]]) -> Tuple[bool, List[str], List[str]]:
        """解析依赖

        Args:
            plugin_name: 插件名称
            dependencies: 依赖列表 [(包名, 版本约束), ...]

        Returns:
            (是否成功, 安装顺序, 错误/警告信息)
        """
        messages: List[str] = []
        to_install: List[str] = []
        graph: Dict[str, Set[str]] = {}

        # 构建依赖图
        for pkg_name, version_constraint in dependencies:
            graph[pkg_name] = set()
            if not self._check_package_satisfies(pkg_name, version_constraint):
                to_install.append(pkg_name)

        # 拓扑排序
        sorted_packages = self._topological_sort(graph)
        if sorted_packages is None:
            return False, [], [f"插件 '{plugin_name}' 存在循环依赖"]

        # 添加主包
        if plugin_name not in sorted_packages:
            sorted_packages.insert(0, plugin_name)

        return True, sorted_packages, messages

    def _check_package_satisfies(self, package_name: str, version_constraint: str) -> bool:
        """检查已安装的包是否满足版本约束"""
        if not version_constraint:
            return package_name in self._installed_packages

        installed_version = self._installed_packages.get(package_name)
        if not installed_version:
            return False

        return self._check_version_compatible(installed_version, version_constraint)

    def _check_version_compatible(self, version: str, constraint: str) -> bool:
        """检查版本兼容性"""
        if not constraint:
            return True

        # 简化实现
        if constraint.startswith(">="):
            required = constraint[2:].strip()
            return version >= required
        elif constraint.startswith("=="):
            required = constraint[2:].strip()
            return version == required

        return True

    def _topological_sort(self, graph: Dict[str, Set[str]]) -> Optional[List[str]]:
        """拓扑排序"""
        in_degree = {node: 0 for node in graph}
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] = in_degree.get(dep, 0)

        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        sorted_list = []

        while queue:
            node = queue.popleft()
            sorted_list.append(node)

            for other in graph:
                if node in graph[other]:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        if len(sorted_list) != len(graph):
            return None

        return sorted_list


# ---------------------------------------------------------------------------
# 插件安装器
# ---------------------------------------------------------------------------

class PluginInstaller:
    """插件安装器

    提供完整的插件安装流程，包括下载、验证、依赖解析、安装和回滚。
    """

    def __init__(self, plugins_dir: Optional[str] = None,
                 backup_dir: Optional[str] = None,
                 signature_verifier: Optional[SignatureVerifier] = None):
        """
        Args:
            plugins_dir: 插件目录
            backup_dir: 备份目录
            signature_verifier: 签名验证器
        """
        self._plugins_dir = plugins_dir or os.path.join(
            os.getcwd(), "plugins"
        )
        self._backup_dir = backup_dir
        self._signature_verifier = signature_verifier or SignatureVerifier()
        self._downloader = PackageDownloader()
        self._dep_resolver = InstallerDependencyResolver()
        self._rollback_manager = RollbackManager(backup_dir)
        self._installed_manifest: Dict[str, dict] = {}
        self._install_hooks: List[Callable[[str], None]] = []

        os.makedirs(self._plugins_dir, exist_ok=True)
        self._load_installed_manifest()

    def add_install_hook(self, hook: Callable[[str], None]) -> None:
        """添加安装钩子"""
        self._install_hooks.append(hook)

    def remove_install_hook(self, hook: Callable[[str], None]) -> None:
        """移除安装钩子"""
        if hook in self._install_hooks:
            self._install_hooks.remove(hook)

    def install(self, package_info: PackageInfo,
               verify_signature: bool = True) -> InstallResult:
        """安装插件

        Args:
            package_info: 包信息
            verify_signature: 是否验证签名

        Returns:
            安装结果
        """
        start_time = time.time()
        result = InstallResult(
            success=False,
            plugin_name=package_info.name,
        )

        # 下载包
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, f"{package_info.name}.zip")

            download_success, error = self._downloader.download(
                package_info.url,
                archive_path,
                expected_checksum=package_info.checksum or None,
            )

            if not download_success:
                result.errors.append(f"下载失败: {error}")
                return result

            # 验证签名
            if verify_signature and package_info.signature_url:
                verified, error = self._signature_verifier.verify_package(
                    archive_path, package_info
                )
                if not verified:
                    result.errors.append(f"签名验证失败: {error}")
                    return result

            # 解压包
            extract_dir = os.path.join(tmpdir, "extracted")
            success, error, extracted_files = self._downloader.extract_package(
                archive_path, extract_dir
            )
            if not success:
                result.errors.append(f"解压失败: {error}")
                return result

            # 查找插件规范文件
            manifest = self._find_plugin_manifest(extract_dir)
            if not manifest:
                result.errors.append("未找到插件清单文件 (plugin.json 或 plugin.yaml)")
                return result

            # 验证依赖
            deps = [(d.name, d.version) for d in manifest.spec.dependencies]
            dep_success, install_order, dep_messages = self._dep_resolver.resolve_dependencies(
                manifest.spec.name, deps
            )
            if not dep_success:
                result.errors.extend(dep_messages)
                return result

            # 安装Python依赖
            for dep in manifest.spec.dependencies:
                if not self._install_python_package(dep.name, dep.version):
                    result.errors.append(f"安装依赖失败: {dep.name}")
                    self._rollback_manager.rollback()
                    return result

            # 安装插件文件
            plugin_dir = os.path.join(self._plugins_dir, manifest.spec.name)
            try:
                # 备份已存在的插件
                if os.path.exists(plugin_dir):
                    self._rollback_manager.record_directory_remove(plugin_dir)

                shutil.copytree(extract_dir, plugin_dir)
                result.installed_files.append(plugin_dir)

            except (IOError, OSError, shutil.Error) as e:
                result.errors.append(f"复制文件失败: {str(e)}")
                self._rollback_manager.rollback()
                return result

            # 执行安装后钩子
            for hook in self._install_hooks:
                try:
                    hook(manifest.spec.name)
                except Exception as e:
                    result.warnings.append(f"安装钩子执行失败: {str(e)}")

            # 更新已安装清单
            self._update_installed_manifest(manifest)
            self._rollback_manager.clear()

            result.success = True
            result.message = f"插件 '{manifest.spec.name}' 安装成功"

        result.duration = time.time() - start_time
        return result

    def uninstall(self, plugin_name: str) -> InstallResult:
        """卸载插件

        Args:
            plugin_name: 插件名称

        Returns:
            卸载结果
        """
        start_time = time.time()
        result = InstallResult(
            success=False,
            plugin_name=plugin_name,
        )

        plugin_dir = os.path.join(self._plugins_dir, plugin_name)
        if not os.path.exists(plugin_dir):
            result.errors.append(f"插件不存在: {plugin_name}")
            return result

        # 备份以支持回滚
        backup_path = self._rollback_manager.record_file_backup(plugin_dir)

        try:
            # 读取插件信息获取依赖
            manifest_path = os.path.join(plugin_dir, "plugin.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    plugin_data = json.load(f)
                    # 注意：这里不自动卸载依赖，因为可能其他插件也在使用

            # 删除插件目录
            shutil.rmtree(plugin_dir)
            result.installed_files.append(plugin_dir)

            # 从已安装清单中移除
            if plugin_name in self._installed_manifest:
                del self._installed_manifest[plugin_name]
                self._save_installed_manifest()

            result.success = True
            result.message = f"插件 '{plugin_name}' 卸载成功"

        except (IOError, OSError, shutil.Error) as e:
            result.errors.append(f"卸载失败: {str(e)}")
            # 回滚
            if backup_path and os.path.exists(backup_path):
                shutil.copytree(backup_path, plugin_dir)
            self._rollback_manager.clear()
            return result

        result.duration = time.time() - start_time
        return result

    def update(self, plugin_name: str, new_package_info: PackageInfo) -> InstallResult:
        """更新插件

        Args:
            plugin_name: 插件名称
            new_package_info: 新包信息

        Returns:
            更新结果
        """
        # 先卸载
        uninstall_result = self.uninstall(plugin_name)
        if not uninstall_result.success:
            return uninstall_result

        # 再安装
        install_result = self.install(new_package_info)
        if not install_result.success:
            # 尝试回滚
            return install_result

        return install_result

    def rollback_last(self) -> Tuple[bool, List[str]]:
        """回滚最后一次安装

        Returns:
            (是否成功, 错误信息)
        """
        return self._rollback_manager.rollback()

    def _install_python_package(self, package_name: str, version: str = "") -> bool:
        """安装Python包"""
        self._rollback_manager.record_pip_install(package_name, version)

        try:
            cmd = [self._get_python_path(), "-m", "pip", "install"]
            if version:
                cmd.append(f"{package_name}=={version}")
            else:
                cmd.append(package_name)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0

        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False

    def _get_python_path(self) -> str:
        """获取Python解释器路径"""
        return os.environ.get("PYTHON_PATH", "python")

    def _find_plugin_manifest(self, directory: str):
        """查找插件清单文件"""
        from .spec import parse_manifest

        for filename in ["plugin.json", "plugin.yaml", "plugin.yml"]:
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                return parse_manifest(path)

        # 递归搜索
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.startswith("plugin.") and filename.endswith((".json", ".yaml", ".yml")):
                    return parse_manifest(os.path.join(root, filename))

        return None

    def _load_installed_manifest(self) -> None:
        """加载已安装插件清单"""
        manifest_path = os.path.join(self._plugins_dir, ".installed.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    self._installed_manifest = json.load(f)
            except (IOError, json.JSONDecodeError):
                self._installed_manifest = {}

    def _save_installed_manifest(self) -> None:
        """保存已安装插件清单"""
        manifest_path = os.path.join(self._plugins_dir, ".installed.json")
        try:
            with open(manifest_path, "w") as f:
                json.dump(self._installed_manifest, f, indent=2)
        except IOError:
            pass

    def _update_installed_manifest(self, manifest) -> None:
        """更新已安装清单"""
        self._installed_manifest[manifest.spec.name] = {
            "version": manifest.spec.version,
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "path": os.path.join(self._plugins_dir, manifest.spec.name),
            "metadata": manifest.spec.metadata,
        }
        self._save_installed_manifest()

    def get_installed_plugins(self) -> Dict[str, dict]:
        """获取已安装的插件"""
        return self._installed_manifest.copy()

    def is_plugin_installed(self, plugin_name: str) -> bool:
        """检查插件是否已安装"""
        return plugin_name in self._installed_manifest
