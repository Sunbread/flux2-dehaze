from __future__ import annotations

from typing import Any, Mapping

import yaml
from pathlib import Path

from .types import PathInput


def load_config(config_path: PathInput) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: Mapping[str, Any], path: PathInput) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
