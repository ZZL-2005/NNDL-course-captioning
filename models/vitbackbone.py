import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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

        # 1) 先创建 ViT 模型，但不让它联网
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,      # ← 必须为 False！否则会去 HuggingFace 拉权重
            num_classes=0          # ← 去掉头部（我们只要 features）
        )

        # 2) 你的本地权重路径（请确认路径一致）
        ckpt_path = "/home/chenzhican/zhangzilu/hwnndl/model/weights/jx_vit_base_p16_224-80ecf9dd.pth"

        print(f"[INFO] Loading local ViT weights from: {ckpt_path}")

        # 3) 加载 checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # timm 权重格式可能是 {'model': {...}} 或直接 state_dict
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        print("[INFO] Missing keys:", missing)
        print("[INFO] Unexpected keys:", unexpected)

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