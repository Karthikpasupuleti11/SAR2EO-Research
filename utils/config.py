from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    base_path = Path("configs/config.yaml")
    if path.resolve() != base_path.resolve() and base_path.exists():
        with base_path.open("r", encoding="utf-8") as handle:
            base_config = yaml.safe_load(handle)
        config = deep_merge(base_config, config)

    return config
