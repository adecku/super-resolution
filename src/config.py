"""Configuration loading utilities for YAML configs with inheritance."""

from pathlib import Path
import yaml


def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    """Load YAML config with inheritance support."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    if "inherit" in config:
        inherit_path = Path(config.pop("inherit"))
        # If inherit path is absolute, use it directly
        # If it starts with "configs/", resolve from current working directory
        # Otherwise, resolve relative to current config's directory
        if inherit_path.is_absolute():
            base_path = inherit_path
        elif str(inherit_path).startswith("configs/"):
            base_path = Path.cwd() / inherit_path
        else:
            base_path = path.parent / inherit_path
        config = _merge_dicts(load_config(base_path), config)
    
    return config


# if __name__ == "__main__":
#     common = load_config("configs/common.yaml")
#     print(f"common: project.name={common['project']['name']}, train.epochs={common['train']['epochs']}")
    
#     srcnn = load_config("configs/train_srcnn_x2.yaml")
#     print(f"srcnn: model.name={srcnn['model']['name']}, data.scale={srcnn['data']['scale']}, model.params={srcnn['model']['params']}")