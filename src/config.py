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
        
        if inherit_path.is_absolute():
            base_path = inherit_path
        else:
            inherit_parts = inherit_path.parts
            parent_name = path.parent.name
            
            if inherit_parts and inherit_parts[0] == parent_name:
                inherit_path = Path(*inherit_parts[1:])
            
            base_path = (path.parent / inherit_path).resolve()
        config = _merge_dicts(load_config(base_path), config)
    
    return config