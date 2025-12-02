#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
适配自定义 TransformerDecoderLayerWithAttn 的可视化工具。
无需使用 hook，因为每层 decoder 已经自己保存了注意力矩阵：

    layer.self_attn_map  : (B, heads, L, L)
    layer.cross_attn_map : (B, heads, L, S)
"""

from typing import Dict, Tuple
import torch
from PIL import Image

@torch.no_grad()
def run_attn(
    model,
    image_path: str,
    transform,
    device: str = "cuda",
) -> Tuple[Image.Image, Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:

    model.eval()
    model.to(device)

    # === 1. 读图 ===
    pil = Image.open(image_path).convert("RGB")
    x = transform(pil).unsqueeze(0).to(device)

    # === 2. 触发模型 forward ===
    _ = model.greedy_decode(x)

    # === 3. 读取注意力 ===
    self_attn = {}
    cross_attn = {}
    norms = {}

    for idx, layer in enumerate(model.decoder.decoder_layers):
        self_attn[idx] = (
            layer.self_attn_map.detach().cpu()
            if layer.self_attn_map is not None else None
        )
        cross_attn[idx] = (
            layer.cross_attn_map.detach().cpu()
            if layer.cross_attn_map is not None else None
        )
        norms[idx] = {
            'sa': layer.sa_update_norm.detach().cpu() if layer.sa_update_norm is not None else None,
            'ca': layer.ca_update_norm.detach().cpu() if layer.ca_update_norm is not None else None,
            'ffn': layer.ffn_update_norm.detach().cpu() if layer.ffn_update_norm is not None else None,
        }

    return pil, self_attn, cross_attn, norms
