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
from tools.functions import collate_lm as collate_fn


# ================================================================
#           可学习 embedding + 一热输入 + Transformer LM
# ================================================================
class AdjustableEmbeddingLM(nn.Module):
    """
    One-hot 输入 + 可调 embedding dim + Transformer Decoder
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 109,      # 可调
        max_len: int = 80,
        n_heads: int = 1,        # 默认 1，避免整除问题
        num_layers: int = 4,
        dim_ff: int = 2048,
        pad_idx: int = 0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.pad_idx = pad_idx

        # 1. one-hot -> emb_dim
        self.input_proj = nn.Linear(vocab_size, emb_dim)

        # 2. 位置编码 emb_dim 维度
        self.pos_emb = nn.Embedding(max_len, emb_dim)

        # 3. Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. output: emb_dim -> vocab_size
        self.output_proj = nn.Linear(emb_dim, vocab_size)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _build_causal_mask(self, L, device):
        mask = torch.full((L, L), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def _build_padding_mask(self, caps):
        return caps == self.pad_idx

    def forward(self, caps):
        B, L = caps.shape
        device = caps.device

        # one-hot
        x = torch.nn.functional.one_hot(caps, num_classes=self.vocab_size).float()
        # one-hot -> emb_dim
        x = self.input_proj(x)

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos_ids)

        tgt_mask = self._build_causal_mask(L, device)
        tgt_pad = self._build_padding_mask(caps)

        memory = torch.zeros(B, 1, self.emb_dim, device=device)
        mem_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)

        h = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad,
        )

        logits = self.output_proj(h)
        return logits


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

    # 基础训练参数
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # 模型结构参数（新增 emb_dim）
    parser.add_argument("--vocab_size", type=int, default=109)
    parser.add_argument("--emb_dim", type=int, default=109)  # 新增可调 embedding 维度
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pad_idx", type=int, default=0)

    # 数据路径
    parser.add_argument("--train_json", type=str, default="/home/chenzhican/zhangzilu/NNDL-course-captioning/data/train.json")
    parser.add_argument("--val_json", type=str, default="/home/chenzhican/zhangzilu/NNDL-course-captioning/data/test.json")
    parser.add_argument("--image_root", type=str, default="/data/zilu/images")
    parser.add_argument("--freeze_input", action="store_true")

    # 输出
    parser.add_argument("--save_dir", type=str, default="outputs_lm/")

    args = parser.parse_args()

    set_seed(args.seed)
    prepare_dirs(args.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 数据
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

    # 模型（用可调 embedding）
    model = AdjustableEmbeddingLM(
    vocab_size=args.vocab_size,
    emb_dim=args.emb_dim,
    max_len=args.max_len,
    n_heads=args.n_heads,
    num_layers=args.num_layers,
    pad_idx=args.pad_idx,
    freeze_input=args.freeze_input
).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)

    # 保存 loss.json
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
