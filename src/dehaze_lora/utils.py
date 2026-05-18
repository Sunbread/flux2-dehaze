import yaml
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str | Path):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
