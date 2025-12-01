"""
Stage 2: 指标计算模块
======================
给定:
    - Stage 1 输出的 JSON 文件 (包含 gt_text, pred_text, loss 等)

输出:
    - 最终 JSON 文件，每条数据新增:
        - metrics.METEOR: METEOR 分数
        - metrics.ROUGE_L: ROUGE-L 分数
        - metrics.CIDEr: CIDEr-D 分数
        - metrics.SPICE: SPICE 分数 (如果 Java 可用)
    - 同时输出整体统计信息
"""

import os
import sys
import json
from tqdm import tqdm
from typing import Optional

# 可选: pycocoevalcap 相关
try:
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    HAS_COCO_EVAL = True
except ImportError:
    HAS_COCO_EVAL = False
    print("[Warning] pycocoevalcap 未安装，部分指标可能不可用")

try:
    from pycocoevalcap.spice.spice import Spice
    HAS_SPICE = True
except ImportError:
    HAS_SPICE = False


def compute_meteor_scores(gts: dict, res: dict) -> dict:
    """计算 METEOR 分数，返回每个样本的分数"""
    if not HAS_COCO_EVAL:
        return {k: -1.0 for k in gts.keys()}
    
    scorer = Meteor()
    avg_score, sample_scores = scorer.compute_score(gts, res)
    return {k: sample_scores[i] for i, k in enumerate(gts.keys())}


def compute_rouge_scores(gts: dict, res: dict) -> dict:
    """计算 ROUGE-L 分数，返回每个样本的分数"""
    if not HAS_COCO_EVAL:
        return {k: -1.0 for k in gts.keys()}
    
    scorer = Rouge()
    avg_score, sample_scores = scorer.compute_score(gts, res)
    return {k: sample_scores[i] for i, k in enumerate(gts.keys())}


def compute_cider_scores(gts: dict, res: dict) -> dict:
    """计算 CIDEr-D 分数，返回每个样本的分数"""
    if not HAS_COCO_EVAL:
        return {k: -1.0 for k in gts.keys()}
    
    scorer = Cider()
    avg_score, sample_scores = scorer.compute_score(gts, res)
    return {k: sample_scores[i] for i, k in enumerate(gts.keys())}


def compute_spice_scores(gts: dict, res: dict) -> dict:
    """计算 SPICE 分数，返回每个样本的分数"""
    if not HAS_SPICE:
        return {k: -1.0 for k in gts.keys()}
    
    try:
        scorer = Spice()
        avg_score, sample_scores = scorer.compute_score(gts, res)
        # SPICE 返回的是 dict 列表，需要提取 'All' 字段的 'f' 值
        scores = {}
        for i, k in enumerate(gts.keys()):
            if isinstance(sample_scores[i], dict):
                scores[k] = sample_scores[i].get('All', {}).get('f', -1.0)
            else:
                scores[k] = sample_scores[i]
        return scores
    except Exception as e:
        print(f"[Warning] SPICE 计算失败: {e}")
        return {k: -1.0 for k in gts.keys()}


def run_stage2_metrics(
    input_json: str,
    output_json: str,
    compute_spice: bool = True,
) -> dict:
    """
    Stage 2: 指标计算
    
    Args:
        input_json: Stage 1 输出的 JSON 文件路径
        output_json: 最终输出的 JSON 文件路径
        compute_spice: 是否计算 SPICE (需要 Java 环境)
        
    Returns:
        dict: 包含整体统计信息的字典
            - total_samples: 总样本数
            - avg_metrics: 各指标的平均值
            - output_path: 输出文件路径
    """
    # 加载 Stage 1 结果
    print(f"[Stage2] 加载输入文件: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"[Stage2] 总样本数: {total_samples}")
    
    # 构建 gts 和 res 字典 (pycocoevalcap 格式)
    # 格式: {id: [sentence]}
    gts = {}
    res = {}
    for i, item in enumerate(data):
        gts[i] = [item["gt_text"]]
        res[i] = [item["pred_text"]]
    
    # 计算各指标
    print("[Stage2] 计算 METEOR...")
    meteor_scores = compute_meteor_scores(gts, res)
    
    print("[Stage2] 计算 ROUGE-L...")
    rouge_scores = compute_rouge_scores(gts, res)
    
    print("[Stage2] 计算 CIDEr-D...")
    cider_scores = compute_cider_scores(gts, res)
    
    spice_scores = {}
    if compute_spice:
        print("[Stage2] 计算 SPICE...")
        spice_scores = compute_spice_scores(gts, res)
    else:
        spice_scores = {k: -1.0 for k in gts.keys()}
    
    # 将指标写入每条数据
    for i, item in enumerate(data):
        item["metrics"] = {
            "METEOR": round(meteor_scores.get(i, -1.0), 6),
            "ROUGE_L": round(rouge_scores.get(i, -1.0), 6),
            "CIDEr": round(cider_scores.get(i, -1.0), 6),
            "SPICE": round(spice_scores.get(i, -1.0), 6) if spice_scores.get(i, -1.0) >= 0 else -1.0,
        }
    
    # 计算整体平均
    avg_metrics = {
        "METEOR": sum(meteor_scores.values()) / total_samples if total_samples > 0 else 0,
        "ROUGE_L": sum(rouge_scores.values()) / total_samples if total_samples > 0 else 0,
        "CIDEr": sum(cider_scores.values()) / total_samples if total_samples > 0 else 0,
        "SPICE": sum(v for v in spice_scores.values() if v >= 0) / max(1, sum(1 for v in spice_scores.values() if v >= 0)),
    }
    avg_loss = sum(item["loss"] for item in data) / total_samples if total_samples > 0 else 0
    
    # 保存结果
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    
    # 最终输出包含数据和汇总
    final_output = {
        "summary": {
            "total_samples": total_samples,
            "avg_loss": round(avg_loss, 6),
            "avg_metrics": {k: round(v, 6) for k, v in avg_metrics.items()},
        },
        "samples": data
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Stage2 完成]")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 平均 Loss: {avg_loss:.4f}")
    print(f"  - 平均指标:")
    for k, v in avg_metrics.items():
        print(f"      {k}: {v:.4f}")
    print(f"  - 输出文件: {output_json}")
    
    return {
        "total_samples": total_samples,
        "avg_loss": avg_loss,
        "avg_metrics": avg_metrics,
        "output_path": output_json
    }


# ============ 命令行接口 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 2: 指标计算")
    parser.add_argument("--input_json", type=str, required=True, help="Stage 1 输出的 JSON 路径")
    parser.add_argument("--output_json", type=str, required=True, help="最终输出 JSON 路径")
    parser.add_argument("--no_spice", action="store_true", help="跳过 SPICE 计算 (无需 Java)")
    
    args = parser.parse_args()
    
    run_stage2_metrics(
        input_json=args.input_json,
        output_json=args.output_json,
        compute_spice=not args.no_spice,
    )
