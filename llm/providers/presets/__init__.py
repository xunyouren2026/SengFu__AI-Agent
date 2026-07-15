"""
预置模型配置包

包含各国产模型提供商的预置配置文件。

可用的预置配置:
- china_models: 国产模型配置 (30+模型)

Author: AGI Team
Version: 1.0.0
"""

from pathlib import Path

PRESETS_DIR = Path(__file__).parent


def get_preset_path(preset_name: str) -> Path:
    """
    获取预置配置文件路径

    Args:
        preset_name: 预置配置名称

    Returns:
        配置文件路径

    Raises:
        FileNotFoundError: 预置配置不存在
    """
    # 尝试YAML格式
    yaml_path = PRESETS_DIR / f"{preset_name}.yaml"
    if yaml_path.exists():
        return yaml_path

    # 尝试JSON格式
    json_path = PRESETS_DIR / f"{preset_name}.json"
    if json_path.exists():
        return json_path

    raise FileNotFoundError(f"预置配置不存在: {preset_name}")


def list_presets() -> list:
    """
    列出所有可用的预置配置

    Returns:
        预置配置名称列表
    """
    presets = []
    for f in PRESETS_DIR.glob("*.yaml"):
        presets.append(f.stem)
    for f in PRESETS_DIR.glob("*.json"):
        if f.stem not in presets:
            presets.append(f.stem)
    return sorted(presets)


__all__ = [
    "PRESETS_DIR",
    "get_preset_path",
    "list_presets",
]
