"""
Task 6 (Pre-Norm): ViT Encoder-Decoder Image Captioning Training Script
=====================================================================

- Model: ViT Encoder + Pre-Norm Transformer Decoder
- Logs: numeric train / test loss (epoch 0 included)
- Checkpoints: configurable save interval
"""

import os
import sys
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# ============ 路径设置 ============
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.dataset import CaptionDataset
from tools.functions import collate_fn
from models.vit_encoder_decoder_prenorm import ImageCaptionModel


# ================================================================
#                           配置参数
# ================================================================

CONFIG = {
    # ---------- 训练参数 ----------
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "seed": 42,

    # ---------- 模型参数 ----------
    "vocab_size": 109,
    "d_model": 512,
    "n_heads": 8,
    "num_layers": 4,
    "max_len": 128,

    # ---------- 特殊 token ----------
    "pad_idx": 0,
    "start_idx": 1,
    "end_idx": 2,

    # ---------- 数据路径 ----------
    "train_json": "/home/chenzhican/zhangzilu/NNDL-course-captioning/data/train.json",
    "test_json": "/home/chenzhican/zhangzilu/NNDL-course-captioning/data/test.json",   # ★ 语义上视为 test
    "image_root": "/data/zilu/images",

    # ---------- 输出路径 ----------
    "save_dir": "outputs_prenorm_layer_1",

    # ---------- checkpoint ----------
    "save_every": 1,   # 每多少个 epoch 存一次权重
}


# ================================================================
#                           工具函数
# ================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_output_dirs(save_dir: str):
    os.makedirs(os.path.join(save_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test_results"), exist_ok=True)


# ================================================================
#                     Train / Test 核心逻辑
# ================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", ncols=100)
    for imgs, caps, lengths, names in pbar:
        imgs = imgs.to(device)
        caps = caps.to(device)

        optimizer.zero_grad()
        logits, targets = model(imgs, caps)

        B, L, V = logits.shape
        loss = criterion(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, epoch, split: str):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{split}]", ncols=100)
    for imgs, caps, lengths, names in pbar:
        imgs = imgs.to(device)
        caps = caps.to(device)

        logits, targets = model(imgs, caps)
        B, L, V = logits.shape
        loss = criterion(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
        )

        total_loss += loss.item()

    return total_loss / len(loader)


# ================================================================
#                           主函数
# ================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    set_seed(CONFIG["seed"])
    create_output_dirs(CONFIG["save_dir"])

    # ---------- 数据 ----------
    transform = get_transform()

    train_ds = CaptionDataset(CONFIG["train_json"], CONFIG["image_root"], transform)
    test_ds  = CaptionDataset(CONFIG["test_json"],  CONFIG["image_root"], transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=16,
        collate_fn=collate_fn,
    )

    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Test  samples: {len(test_ds)}")

    # ---------- 模型 ----------
    model = ImageCaptionModel(
        vocab_size=CONFIG["vocab_size"],
        pad_idx=CONFIG["pad_idx"],
        start_idx=CONFIG["start_idx"],
        end_idx=CONFIG["end_idx"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        num_layers=CONFIG["num_layers"],
        max_len=CONFIG["max_len"],
        freeze_encoder=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG["pad_idx"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # ---------- 数值日志 ----------
    loss_log = {
        "train": [],
        "test": [],
    }

    # ============================================================
    #                Epoch 0：未训练模型评估
    # ============================================================

    print("\n[INFO] Epoch 0 (untrained model evaluation)")
    train_loss_0 = evaluate_loss(
        model, train_loader, criterion, device, epoch=0, split="Train"
    )
    test_loss_0 = evaluate_loss(
        model, test_loader, criterion, device, epoch=0, split="Test"
    )

    loss_log["train"].append(train_loss_0)
    loss_log["test"].append(test_loss_0)

    print(f"[Epoch 0] Train Loss: {train_loss_0:.4f}")
    print(f"[Epoch 0] Test  Loss: {test_loss_0:.4f}")

    # ============================================================
    #                        训练循环
    # ============================================================

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        test_loss = evaluate_loss(
            model, test_loader, criterion, device, epoch, split="Test"
        )

        loss_log["train"].append(train_loss)
        loss_log["test"].append(test_loss)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
        print(f"[Epoch {epoch}] Test  Loss: {test_loss:.4f}")

        # ---------- 保存 checkpoint ----------
        if epoch % CONFIG["save_every"] == 0:
            ckpt_path = os.path.join(
                CONFIG["save_dir"], "ckpts", f"epoch{epoch}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Saved] {ckpt_path}")

        # ---------- 保存 loss 数值日志 ----------
        log_path = os.path.join(CONFIG["save_dir"], "logs", "loss_log.json")
        with open(log_path, "w") as f:
            json.dump(loss_log, f, indent=2)

    print("\n[INFO] Training finished!")


if __name__ == "__main__":
    main()
