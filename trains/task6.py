# train/train.py
import os
import sys
import json
import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.dataset import CaptionDataset
from tools.functions import collate_fn
from models.vit_encoder_decoder import ImageCaptionModel


############################################
# ========== 0. 固定随机种子 ===============
############################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


############################################
# ========== 1. 超参数 =====================
############################################

CONFIG = {
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-4,

    "max_len": 128,
    "vocab_size": 109,
    "pad_idx": 0,
    "start_idx": 1,
    "end_idx": 2,

    "d_model": 512,
    "n_heads": 8,
    "num_layers": 4,

    "train_json": "processed/train.json",
    "val_json": "processed/val.json",
    "image_root": "/data/zilu/images",

    "save_dir": "outputs/",
    "seed": 42,
}

os.makedirs(os.path.join(CONFIG["save_dir"], "ckpts"), exist_ok=True)
os.makedirs(os.path.join(CONFIG["save_dir"], "test_results"), exist_ok=True)
os.makedirs(os.path.join(CONFIG["save_dir"], "logs"), exist_ok=True)


############################################
# ========== 2. 单轮训练 ===================
############################################

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", ncols=100)

    for imgs, caps, lengths, names in pbar:
        imgs = imgs.to(device)
        caps = caps.to(device)

        optimizer.zero_grad()

        logits, targets = model(imgs, caps)

        B, L, V = logits.shape
        loss = criterion(logits.reshape(B*L, V), targets.reshape(B*L))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


############################################
# ========== 3. 验证：只保存 token ==========
############################################

def evaluate_and_save(model, loader, device, epoch, cfg):

    model.eval()
    results = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Test]", ncols=100)

    with torch.no_grad():
        for imgs, caps, lengths, names in pbar:

            imgs = imgs.to(device)

            pred_ids = model.greedy_decode(imgs)   # (B, max_len)
            gts = caps.tolist()
            preds = pred_ids.tolist()

            B = len(names)

            for i in range(B):
                gt_seq = gts[i]
                pred_seq = preds[i]

                # 删除 <START> = 1
                if cfg["start_idx"] in gt_seq:
                    gt_seq = gt_seq[1:]
                if cfg["end_idx"] in gt_seq:
                    gt_seq = gt_seq[:gt_seq.index(cfg["end_idx"])]

                if cfg["start_idx"] in pred_seq:
                    pred_seq = pred_seq[1:]
                if cfg["end_idx"] in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(cfg["end_idx"])]

                results.append({
                    "img": names[i],
                    "gt_ids": gt_seq,
                    "pred_ids": pred_seq
                })

    save_path = os.path.join(
        cfg["save_dir"], "test_results", f"epoch{epoch}_tokens.json"
    )
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Saved] {save_path}")


############################################
# ========== 4. 主训练流程 ==================
############################################

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(CONFIG["seed"])

    # 加载词表（你训练阶段不需要 id2token）
    vocab = json.load(open("processed/vocab.json"))
    id2token = vocab["id2token"]

    # transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_ds = CaptionDataset(CONFIG["train_json"], CONFIG["image_root"], transform)
    val_ds   = CaptionDataset(CONFIG["val_json"],   CONFIG["image_root"], transform)

    train_loader = DataLoader(train_ds,
                              batch_size=CONFIG["batch_size"],
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn)

    val_loader = DataLoader(val_ds,
                            batch_size=CONFIG["batch_size"],
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)

    # Model
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

    # train loop
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        ckpt_path = os.path.join(CONFIG["save_dir"], "ckpts", f"epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Saved] {ckpt_path}")

        evaluate_and_save(model, val_loader, device, epoch, CONFIG)


if __name__ == "__main__":
    main()
