"""
Stage 1: 推理预测模块
======================
给定:
    - 加载好权重的 model
    - 数据集 JSON 文件路径
    - 图片根目录
    - 词表 JSON 文件路径

输出:
    - JSON 文件，每条数据包含:
        - img: 图片名
        - gt_ids: ground truth token ids
        - gt_text: ground truth 文本
        - pred_ids: 预测 token ids  
        - pred_text: 预测文本
        - loss: 该样本的 loss 值
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.dataset import CaptionDataset
from tools.functions import collate_fn


def ids_to_text(ids: list, id2token: dict, start_idx: int = 1, end_idx: int = 2) -> str:
    """
    将 token id 序列转换为文本
    自动去除 <START>, <END>, <PAD> 等特殊 token
    """
    tokens = []
    for t in ids:
        if t == start_idx:  # <START>
            continue
        if t == end_idx:    # <END>
            break
        if t == 0:          # <PAD>
            continue
        tokens.append(id2token.get(str(t), f"<UNK:{t}>"))
    return " ".join(tokens)


def compute_sample_loss(
    model: nn.Module,
    images: torch.Tensor,
    captions: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    计算每个样本的 loss (不做 reduction)
    返回: (B,) 每个样本的平均 loss
    """
    logits, targets = model(images, captions)  # (B, L-1, V), (B, L-1)
    B, L, V = logits.shape
    
    # 计算每个 token 的 loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none')
    token_losses = criterion(logits.reshape(B * L, V), targets.reshape(B * L))  # (B*L,)
    token_losses = token_losses.reshape(B, L)  # (B, L)
    
    # 计算每个样本的有效 token 数量，然后求平均
    mask = (targets != pad_idx).float()  # (B, L)
    sample_losses = (token_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B,)
    
    return sample_losses


def run_stage1_prediction(
    model: nn.Module,
    data_json: str,
    image_root: str,
    vocab_json: str,
    output_json: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    pad_idx: int = 0,
    start_idx: int = 1,
    end_idx: int = 2,
    transform = None,
) -> dict:
    """
    Stage 1: 推理预测
    
    Args:
        model: 加载好权重的模型
        data_json: 数据集 JSON 文件路径
        image_root: 图片根目录 (每个人可能不同)
        vocab_json: 词表 JSON 文件路径
        output_json: 输出 JSON 文件路径
        batch_size: 批大小
        num_workers: DataLoader 工作进程数
        device: 设备 ("cuda" / "cpu")
        pad_idx: PAD token id
        start_idx: START token id
        end_idx: END token id
        transform: 图像预处理 transform (如果为 None，使用默认)
        
    Returns:
        dict: 包含统计信息的字典
            - total_samples: 总样本数
            - avg_loss: 平均 loss
            - output_path: 输出文件路径
    """
    # 默认 transform
    if transform is None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # 加载词表
    with open(vocab_json, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id2token = vocab["id2token"]
    
    # 创建数据集和 DataLoader
    dataset = CaptionDataset(data_json, image_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    model = model.to(device)
    model.eval()
    
    results = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="[Stage1] Predicting", ncols=100)
        
        for imgs, caps, lengths, names in pbar:
            imgs = imgs.to(device)
            caps = caps.to(device)
            B = imgs.size(0)
            
            # 1) 计算每个样本的 loss
            sample_losses = compute_sample_loss(model, imgs, caps, pad_idx)  # (B,)
            
            # 2) 贪心解码得到预测结果
            pred_ids = model.greedy_decode(imgs)  # (B, max_len)
            
            # 3) 收集结果
            gt_list = caps.tolist()
            pred_list = pred_ids.tolist()
            loss_list = sample_losses.tolist()
            
            for i in range(B):
                gt_ids = gt_list[i]
                pred_seq = pred_list[i]
                loss_val = loss_list[i]
                
                # 清理 gt_ids: 去除 padding
                gt_ids_clean = [t for t in gt_ids if t != pad_idx]
                # 去除 <START> 和 <END> 用于存储
                gt_ids_stored = gt_ids_clean[1:] if gt_ids_clean and gt_ids_clean[0] == start_idx else gt_ids_clean
                if end_idx in gt_ids_stored:
                    gt_ids_stored = gt_ids_stored[:gt_ids_stored.index(end_idx)]
                
                # 清理 pred_ids
                pred_ids_stored = pred_seq[1:] if pred_seq and pred_seq[0] == start_idx else pred_seq
                if end_idx in pred_ids_stored:
                    pred_ids_stored = pred_ids_stored[:pred_ids_stored.index(end_idx)]
                # 去除 padding
                pred_ids_stored = [t for t in pred_ids_stored if t != pad_idx]
                
                results.append({
                    "img": names[i],
                    "gt_ids": gt_ids_stored,
                    "gt_text": ids_to_text(gt_ids, id2token, start_idx, end_idx),
                    "pred_ids": pred_ids_stored,
                    "pred_text": ids_to_text(pred_seq, id2token, start_idx, end_idx),
                    "loss": round(loss_val, 6)
                })
                
                total_loss += loss_val
                total_samples += 1
            
            pbar.set_postfix(avg_loss=total_loss / total_samples)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    print(f"\n[Stage1 完成]")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 平均 Loss: {avg_loss:.4f}")
    print(f"  - 输出文件: {output_json}")
    
    return {
        "total_samples": total_samples,
        "avg_loss": avg_loss,
        "output_path": output_json
    }


# ============ 命令行接口 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1: 模型推理预测")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data_json", type=str, required=True, help="数据集 JSON 路径")
    parser.add_argument("--image_root", type=str, required=True, help="图片根目录")
    parser.add_argument("--vocab_json", type=str, default="data/vocab.json", help="词表 JSON 路径")
    parser.add_argument("--output_json", type=str, required=True, help="输出 JSON 路径")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    
    # 模型参数 (需要与训练时一致)
    parser.add_argument("--vocab_size", type=int, default=109)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)
    
    args = parser.parse_args()
    
    # 加载模型
    from models.vit_encoder_decoder import ImageCaptionModel
    
    model = ImageCaptionModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        max_len=args.max_len,
    )
    
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    print(f"[INFO] 加载权重: {args.checkpoint}")
    
    # 运行 Stage 1
    run_stage1_prediction(
        model=model,
        data_json=args.data_json,
        image_root=args.image_root,
        vocab_json=args.vocab_json,
        output_json=args.output_json,
        batch_size=args.batch_size,
        device=args.device,
    )
