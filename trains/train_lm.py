#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------- 路径设置 ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.dataset import CaptionDataset
from tools.functions import collate_lm as collate_fn   # LM-only collate
from models.LM import OneHotLanguageModel              # One-hot LM


# ================================================================
#                     工具函数
# ================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dirs(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)


# ================================================================
#                     训练与验证
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="[Train]")

    # collate_lm 返回 (caps, lengths)
    for caps, lengths in pbar:
        caps = caps.to(device)

        input_ids = caps[:, :-1]
        target_ids = caps[:, 1:]

        optimizer.zero_grad()
        logits = model(input_ids)

        B, L, V = logits.shape
        loss = criterion(logits.reshape(B * L, V), target_ids.reshape(B * L))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="[Val]")
        for caps, lengths in pbar:
            caps = caps.to(device)

            input_ids = caps[:, :-1]
            target_ids = caps[:, 1:]

            logits = model(input_ids)

            B, L, V = logits.shape
            loss = criterion(logits.reshape(B * L, V), target_ids.reshape(B * L))

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# ================================================================
#                     主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser()

    # ---------------- 基础训练参数 ----------------
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # ---------------- 模型结构参数 ----------------
    parser.add_argument("--vocab_size", type=int, default=109)
    parser.add_argument("--n_heads", type=int, default=1)  # one-hot 必须为 1
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pad_idx", type=int, default=0)

    # ---------------- 数据路径 ----------------
    parser.add_argument("--train_json", type=str, default="/home/chenzhican/zhangzilu/NNDL-course-captioning/data/train.json")
    parser.add_argument("--val_json", type=str, default="/home/chenzhican/zhangzilu/NNDL-course-captioning/data/test.json")
    parser.add_argument("--image_root", type=str, default="/data/zilu/images")

    # ---------------- 输出路径 ----------------
    parser.add_argument("--save_dir", type=str, default="outputs_lm/")

    args = parser.parse_args()

    # ---------------- 初始化 ----------------
    set_seed(args.seed)
    prepare_dirs(args.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ---------------- 数据 ----------------
    train_ds = CaptionDataset(args.train_json, args.image_root, transform=None)
    val_ds = CaptionDataset(args.val_json, args.image_root, transform=None)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )

    # ---------------- 模型构建 ----------------
    # One-hot LM：无需 d_model，内部使用 d_model=vocab_size
    model = OneHotLanguageModel(
        vocab_size=args.vocab_size,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        max_len=args.max_len,
        pad_idx=args.pad_idx
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)

    # ---------------- 只保存一个 loss.json ----------------
    loss_log = []

    print(f"\n[INFO] Start training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\n========== Epoch {epoch} ==========")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        loss_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        with open(os.path.join(args.save_dir, "logs", "loss.json"), "w") as f:
            json.dump(loss_log, f, indent=2)

    print("\n[INFO] Training finished!")


if __name__ == "__main__":
    main()
