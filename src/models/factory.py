"""Model factory for super-resolution architectures."""

import torch

from src.models.srcnn import SRCNN
from src.models.edsr import EDSR
from src.models.swinir import SwinIR


def create_model(model_name: str, cfg: dict, device: torch.device):
    """Create and move model to target device based on config."""
    scale = cfg["data"]["scale"]
    params = cfg["model"].get("params", {})
    model_name = model_name.lower()

    if model_name == "srcnn":
        channels = params.get("channels", 64)
        model = SRCNN(scale=scale, channels=channels)
    elif model_name == "edsr":
        num_feats = params.get("num_feats", 64)
        num_blocks = params.get("num_blocks", 16)
        res_scale = params.get("res_scale", 0.1)
        model = EDSR(scale=scale, num_feats=num_feats, num_blocks=num_blocks, res_scale=res_scale)
    elif model_name == "swinir":
        embed_dim = params.get("embed_dim", 96)
        depths = params.get("depths", [6, 6, 6, 6])
        num_heads = params.get("num_heads", [6, 6, 6, 6])
        window_size = params.get("window_size", 8)
        mlp_ratio = params.get("mlp_ratio", 4.0)
        qkv_bias = params.get("qkv_bias", True)
        drop_rate = params.get("drop_rate", 0.0)
        attn_drop_rate = params.get("attn_drop_rate", 0.0)
        drop_path_rate = params.get("drop_path_rate", 0.1)
        model = SwinIR(
            scale=scale,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
    else:
        raise ValueError(f"Unsupported model: '{model_name}'. Supported models: 'srcnn', 'edsr', 'swinir'")

    return model.to(device)
