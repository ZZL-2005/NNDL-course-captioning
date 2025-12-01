"""
Task 6: ViT Encoder-Decoder 图像描述生成训练脚本
================================================

模型: ViT (预训练) + Transformer Decoder
数据: 服装图像描述数据集
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
from models.vit_encoder_decoder import ImageCaptionModel


# ================================================================
#                           配置参数
# ================================================================

CONFIG = {
    # ---------- 训练参数 ----------
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-4,
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
    "train_json": "processed/train.json",
    "val_json": "processed/val.json",
    "image_root": "/data/zilu/images",
    
    # ---------- 输出路径 ----------
    "save_dir": "outputs/",
}


# ================================================================
#                           工具函数
# ================================================================

def set_seed(seed: int = 42):
    """固定随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform():
    """获取图像预处理 transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_output_dirs(save_dir: str):
    """创建输出目录"""
    os.makedirs(os.path.join(save_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test_results"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)


def clean_sequence(seq: list, start_idx: int, end_idx: int) -> list:
    """清理序列：去除 <START> 和 <END> 之后的部分"""
    # 去除 <START>
    if seq and seq[0] == start_idx:
        seq = seq[1:]
    # 截断到 <END>
    if end_idx in seq:
        seq = seq[:seq.index(end_idx)]
    return seq


# ================================================================
#                           训练与验证
# ================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int) -> float:
    """
    单轮训练
    
    Returns:
        avg_loss: 平均训练损失
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", ncols=100)
    for imgs, caps, lengths, names in pbar:
        imgs = imgs.to(device)
        caps = caps.to(device)
        
        optimizer.zero_grad()
        logits, targets = model(imgs, caps)
        
        B, L, V = logits.shape
        loss = criterion(logits.reshape(B * L, V), targets.reshape(B * L))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(loader)


def evaluate_and_save(model, loader, device, epoch: int, cfg: dict):
    """
    验证并保存预测结果 (token 序列)
    """
    model.eval()
    results = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", ncols=100)
    with torch.no_grad():
        for imgs, caps, lengths, names in pbar:
            imgs = imgs.to(device)
            
            # 贪心解码
            pred_ids = model.greedy_decode(imgs)
            
            # 收集结果
            gts = caps.tolist()
            preds = pred_ids.tolist()
            
            for i in range(len(names)):
                gt_seq = clean_sequence(gts[i], cfg["start_idx"], cfg["end_idx"])
                pred_seq = clean_sequence(preds[i], cfg["start_idx"], cfg["end_idx"])
                
                results.append({
                    "img": names[i],
                    "gt_ids": gt_seq,
                    "pred_ids": pred_seq
                })
    
    # 保存结果
    save_path = os.path.join(cfg["save_dir"], "test_results", f"epoch{epoch}_tokens.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {save_path}")


# ================================================================
#                           主函数
# ================================================================

def main():
    # ---------- 初始化 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    set_seed(CONFIG["seed"])
    create_output_dirs(CONFIG["save_dir"])
    
    # ---------- 数据准备 ----------
    transform = get_transform()
    
    train_ds = CaptionDataset(CONFIG["train_json"], CONFIG["image_root"], transform)
    val_ds = CaptionDataset(CONFIG["val_json"], CONFIG["image_root"], transform)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # ---------- 模型构建 ----------
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
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    
    # ---------- 训练循环 ----------
    print(f"\n[INFO] Start training for {CONFIG['epochs']} epochs...")
    print("=" * 60)
    
    for epoch in range(CONFIG["epochs"]):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
        
        # 保存检查点
        ckpt_path = os.path.join(CONFIG["save_dir"], "ckpts", f"epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Saved] {ckpt_path}")
        
        # 验证
        evaluate_and_save(model, val_loader, device, epoch, CONFIG)
        print("-" * 60)
    
    print("\n[INFO] Training finished!")


if __name__ == "__main__":
    main()
