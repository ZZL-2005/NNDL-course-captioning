"""
评测模块
========

提供两阶段评测体系:
    - Stage 1 (stage1_predict.py): 模型推理，生成预测结果和 loss
    - Stage 2 (stage2_metrics.py): 计算 METEOR/ROUGE-L/CIDEr/SPICE 指标
    - evaluate.py: 统一入口，一键完成全部评测

使用示例:
---------

方式1: 使用统一接口
    from eval.evaluate import run_full_evaluation
    from models.vit_encoder_decoder import ImageCaptionModel
    
    # 加载模型
    model = ImageCaptionModel(vocab_size=109, ...)
    model.load_state_dict(torch.load("checkpoint.pth"))
    
    # 一键评测
    results = run_full_evaluation(
        model=model,
        data_json="data/val.json",
        image_root="/your/image/path",
    )

方式2: 分阶段运行
    from eval.stage1_predict import run_stage1_prediction
    from eval.stage2_metrics import run_stage2_metrics
    
    # Stage 1
    run_stage1_prediction(model, data_json, image_root, vocab_json, output_json)
    
    # Stage 2
    run_stage2_metrics(input_json, output_json)

命令行:
-------
    # 完整评测
    python -m eval.evaluate --checkpoint ckpt.pth --data_json data/val.json --image_root /path/to/images
    
    # 仅 Stage 1
    python -m eval.stage1_predict --checkpoint ckpt.pth --data_json data/val.json --image_root /path/to/images --output_json stage1.json
    
    # 仅 Stage 2
    python -m eval.stage2_metrics --input_json stage1.json --output_json final.json
"""

from eval.stage1_predict import run_stage1_prediction, ids_to_text
from eval.stage2_metrics import run_stage2_metrics
from eval.evaluate import run_full_evaluation

__all__ = [
    "run_stage1_prediction",
    "run_stage2_metrics", 
    "run_full_evaluation",
    "ids_to_text",
]
