"""
统一评测入口
=============
提供一键运行完整评测流程的接口

使用方式:
    1. 作为模块导入:
        from eval.evaluate import run_full_evaluation
        
        results = run_full_evaluation(
            model=model,
            data_json="data/val.json",
            image_root="/your/image/path",
            vocab_json="data/vocab.json",
            output_dir="outputs/eval_results",
        )
    
    2. 命令行:
        python eval/evaluate.py \
            --checkpoint outputs/ckpts/epoch10.pth \
            --data_json data/val.json \
            --image_root /your/image/path \
            --output_dir outputs/eval_results
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.stage1_predict import run_stage1_prediction
from eval.stage2_metrics import run_stage2_metrics


def run_full_evaluation(
    model,
    data_json: str,
    image_root: str,
    vocab_json: str = "data/vocab.json",
    output_dir: str = "outputs/eval_results",
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    compute_spice: bool = True,
    experiment_name: Optional[str] = None,
    transform = None,
) -> dict:
    """
    一键运行完整评测流程
    
    Args:
        model: 加载好权重的模型
        data_json: 数据集 JSON 文件路径
        image_root: 图片根目录
        vocab_json: 词表 JSON 文件路径
        output_dir: 输出目录
        batch_size: 批大小
        num_workers: DataLoader 工作进程数
        device: 设备
        compute_spice: 是否计算 SPICE
        experiment_name: 实验名称 (用于区分不同实验的输出)
        transform: 图像预处理 transform
        
    Returns:
        dict: 完整的评测结果
    """
    # 生成时间戳和实验名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name or f"eval_{timestamp}"
    
    # 创建输出目录
    exp_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # 定义中间文件和最终文件路径
    stage1_output = os.path.join(exp_output_dir, "stage1_predictions.json")
    stage2_output = os.path.join(exp_output_dir, "final_results.json")
    
    print("=" * 60)
    print(f"开始评测: {exp_name}")
    print("=" * 60)
    
    # Stage 1: 推理预测
    print("\n" + "=" * 60)
    print("Stage 1: 模型推理预测")
    print("=" * 60)
    stage1_results = run_stage1_prediction(
        model=model,
        data_json=data_json,
        image_root=image_root,
        vocab_json=vocab_json,
        output_json=stage1_output,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        transform=transform,
    )
    
    # Stage 2: 指标计算
    print("\n" + "=" * 60)
    print("Stage 2: 指标计算")
    print("=" * 60)
    stage2_results = run_stage2_metrics(
        input_json=stage1_output,
        output_json=stage2_output,
        compute_spice=compute_spice,
    )
    
    # 保存评测配置
    config = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "data_json": data_json,
        "image_root": image_root,
        "vocab_json": vocab_json,
        "batch_size": batch_size,
        "device": device,
        "compute_spice": compute_spice,
    }
    config_path = os.path.join(exp_output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("评测完成!")
    print("=" * 60)
    print(f"  - Stage 1 输出: {stage1_output}")
    print(f"  - 最终结果: {stage2_output}")
    print(f"  - 配置文件: {config_path}")
    
    return {
        "experiment_name": exp_name,
        "stage1": stage1_results,
        "stage2": stage2_results,
        "output_dir": exp_output_dir,
    }


# ============ 命令行接口 ============
if __name__ == "__main__":
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description="统一评测入口")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data_json", type=str, required=True, help="数据集 JSON 路径")
    parser.add_argument("--image_root", type=str, required=True, help="图片根目录")
    parser.add_argument("--vocab_json", type=str, default="data/vocab.json", help="词表 JSON 路径")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_spice", action="store_true", help="跳过 SPICE 计算")
    parser.add_argument("--experiment_name", type=str, default=None, help="实验名称")
    
    # 模型参数
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
    
    # 运行评测
    run_full_evaluation(
        model=model,
        data_json=args.data_json,
        image_root=args.image_root,
        vocab_json=args.vocab_json,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        compute_spice=not args.no_spice,
        experiment_name=args.experiment_name,
    )
