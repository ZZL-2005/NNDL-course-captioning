import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os


def _find_vit_ckpt_path() -> str | None:
    """Return a local checkpoint path if configured and exists, else None."""
    # 1) explicit env var wins
    env_path = os.environ.get("VIT_CKPT_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2) try common relative locations from repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(repo_root, "weights", "jx_vit_base_p16_224-80ecf9dd.pth"),
        os.path.join(repo_root, "model", "weights", "jx_vit_base_p16_224-80ecf9dd.pth"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    return None

class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder:
    使用 timm 的 vit_base_patch16_224
    - 不联网加载 (pretrained=False)
    - 手动加载本地权重
    - 去掉分类头
    """

    def __init__(self, d_model=512, num_img_tokens=None, freeze=False):
        super().__init__()

        # Prefer a local checkpoint if available; otherwise fall back to timm pretrained weights.
        ckpt_path = _find_vit_ckpt_path()
        use_pretrained = ckpt_path is None

        # 1) 创建 ViT 模型
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=use_pretrained,
            num_classes=0,
        )

        # 2) 如果有本地权重，则手动加载（并覆盖 timm 初始化权重）
        if ckpt_path is not None:
            print(f"[INFO] Loading local ViT weights from: {ckpt_path}")

            state_dict = torch.load(ckpt_path, map_location="cpu")
            # timm 权重格式可能是 {'model': {...}} 或直接 state_dict
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
            print("[INFO] Missing keys:", missing)
            print("[INFO] Unexpected keys:", unexpected)
        else:
            print("[INFO] No local ViT checkpoint found; using timm pretrained weights.")
            print("[INFO] Tip: set env var VIT_CKPT_PATH to a local .pth to avoid downloads.")

        # 4) 冻结 ViT（可选）
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        # 5) 投影到 d_model
        self.proj = nn.Linear(768, d_model)
        self.num_img_tokens = num_img_tokens

    def forward(self, images):
        feats = self.vit.forward_features(images)  # (B, 197, 768)
        patch_tokens = feats[:, 1:]               # 去掉 CLS → (B, 196, 768)

        x = self.proj(patch_tokens)               # (B, 196, d_model)

        # 可选：把 196 个 token 压缩到 num_img_tokens
        if self.num_img_tokens is not None and self.num_img_tokens < x.size(1):
            B, S, C = x.shape
            x = x.transpose(1, 2)                      # (B, C, S)
            x = F.adaptive_avg_pool1d(x, self.num_img_tokens)  # (B, C, T)
            x = x.transpose(1, 2)                      # (B, T, C)

        return x