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
        inherit_path_str = config.pop("inherit")
        inherit_path = Path(inherit_path_str)
        
        # If inherit path is absolute, use it directly
        if inherit_path.is_absolute():
            base_path = inherit_path
        else:
            # Handle relative paths
            # If inherit_path starts with the same directory name as path.parent,
            # remove that prefix to avoid duplication (e.g., "configs/common.yaml" 
            # when already in "configs/" directory)
            inherit_parts = inherit_path.parts
            parent_name = path.parent.name
            
            if inherit_parts and inherit_parts[0] == parent_name:
                # Remove the duplicate directory name
                inherit_path = Path(*inherit_parts[1:])
            
            # Resolve relative to current config's directory
            base_path = (path.parent / inherit_path).resolve()
        config = _merge_dicts(load_config(base_path), config)
    
    return config


# if __name__ == "__main__":
#     common = load_config("configs/common.yaml")
#     print(f"common: project.name={common['project']['name']}, train.epochs={common['train']['epochs']}")
    
#     srcnn = load_config("configs/train_srcnn_x2.yaml")
#     print(f"srcnn: model.name={srcnn['model']['name']}, data.scale={srcnn['data']['scale']}, model.params={srcnn['model']['params']}")