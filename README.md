# NNDL-course-captioning

> 神经网络与深度学习课程项目：基于 ViT + Transformer 的服装图像描述生成

## 📋 项目简介

本项目实现了一个**图像描述生成 (Image Captioning)** 系统，输入一张服装图片，自动生成描述图中人物穿着的文本。

**模型架构：**
```
图像 (224×224) → ViT Encoder (预训练) → 图像特征 → Transformer Decoder → 文本描述
```

**示例输出：**
```
输入: 一张女性穿着毛衣的图片
输出: "the sweater this lady wears has long sleeves , its fabric is cotton , and it has pure color patterns ."
```

---

## 📁 项目结构

```
NNDL-course-captioning/
├── data/                       # 数据目录
│   ├── train.json              # 训练集 (34035 样本)
│   ├── val.json                # 验证集 (4254 样本)
│   ├── test.json               # 测试集 (4255 样本)
│   ├── vocab.json              # 词表 (109 tokens)
│   ├── captions.json           # 原始描述数据
│   └── preprocess.py           # 数据预处理脚本
│
├── models/                     # 模型定义
│   ├── vit_encoder_decoder.py  # 主模型：ViT Encoder + Transformer Decoder
│   └── vitbackbone.py          # ViT 骨干网络
│
├── trains/                     # 训练脚本
│   └── task6.py                # 训练入口
│
├── eval/                       # 评测模块 ⭐
│   ├── __init__.py             # 模块入口
│   ├── stage1_predict.py       # Stage 1: 推理预测 + Loss 计算
│   ├── stage2_metrics.py       # Stage 2: 指标计算 (METEOR/ROUGE/CIDEr/SPICE)
│   └── evaluate.py             # 统一评测入口
│
├── tools/                      # 工具函数
│   ├── dataset.py              # PyTorch Dataset 定义
│   ├── functions.py            # collate_fn 等工具函数
│   ├── token2id.py             # token → id 转换
│   └── id2token.py             # id → token 转换
│
├── inference/                  # 推理模块 ⭐
│   └── infer.py                # ImageCaptioner 推理类
│
├── experiments/                # 实验分析
│   └── analysis1.ipynb         # 分析 notebook
│
└── outputs/                    # 输出目录 (训练时生成)
    ├── ckpts/                  # 模型检查点
    ├── test_results/           # 测试结果
    └── eval_results/           # 评测结果
```

---

## 📊 数据格式说明

### 1. 数据集 JSON (`train.json` / `val.json` / `test.json`)

每条数据包含图片路径、token ID 序列和序列长度：

```json
{
  "img": "WOMEN-Jackets_Coats-id_00007765-03_2_side.jpg",
  "cap_ids": [1, 3, 35, 30, 99, 32, 15, 8, 21, ..., 2],
  "length": 33
}
```

| 字段 | 说明 |
|------|------|
| `img` | 图片文件名 |
| `cap_ids` | token ID 序列，以 `<START>=1` 开头，`<END>=2` 结尾 |
| `length` | 序列长度 (含 START 和 END) |

### 2. 词表 JSON (`vocab.json`)

包含 109 个 tokens，涵盖服装相关词汇：

```json
{
  "token2id": {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "the": 3,
    "sweater": 35,
    "cotton": 12,
    ...
  },
  "id2token": {
    "0": "<PAD>",
    "1": "<START>",
    "2": "<END>",
    "3": "the",
    ...
  },
  "freq": {
    "the": 121842,
    "is": 118379,
    ...
  }
}
```

**特殊 Token：**
| Token | ID | 说明 |
|-------|-----|------|
| `<PAD>` | 0 | 填充符 |
| `<START>` | 1 | 序列开始 |
| `<END>` | 2 | 序列结束 |

---

## 🚀 快速开始

### 1. 环境要求

```bash
pip install torch torchvision tqdm
pip install pycocoevalcap  # 评测指标 (可选)
```

### 2. 训练模型

```bash
# 修改 trains/task6.py 中的 image_root 为你的图片路径
python trains/task6.py
```

**主要超参数：**
| 参数 | 值 |
|------|-----|
| epochs | 20 |
| batch_size | 32 |
| learning_rate | 1e-4 |
| d_model | 512 |
| n_heads | 8 |
| num_layers | 4 |

### 3. 评测模型

**方式一：Python 代码**
```python
from eval.evaluate import run_full_evaluation
from models.vit_encoder_decoder import ImageCaptionModel
import torch

# 加载模型
model = ImageCaptionModel(vocab_size=109)
model.load_state_dict(torch.load("outputs/ckpts/epoch19.pth"))

# 一键评测
results = run_full_evaluation(
    model=model,
    data_json="data/val.json",
    image_root="/your/image/path",  # 👈 修改为你的图片路径
    output_dir="outputs/eval_results",
)
```

**方式二：命令行**
```bash
python -m eval.evaluate \
    --checkpoint outputs/ckpts/epoch19.pth \
    --data_json data/val.json \
    --image_root /your/image/path \
    --output_dir outputs/eval_results
```

---

## 📈 评测体系

评测分为两个阶段：

### Stage 1: 推理预测 (`stage1_predict.py`)

- 输入：模型 + 数据集 + 图片路径
- 输出：每个样本的 gt_text、pred_text、loss

```json
{
  "img": "xxx.jpg",
  "gt_ids": [3, 19, 20, ...],
  "gt_text": "the tank top this female wears ...",
  "pred_ids": [3, 35, 30, ...],
  "pred_text": "the sweater this ...",
  "loss": 0.123456
}
```

### Stage 2: 指标计算 (`stage2_metrics.py`)

- 输入：Stage 1 的输出
- 输出：每个样本的四个指标 + 整体统计

```json
{
  "summary": {
    "total_samples": 4254,
    "avg_loss": 0.5234,
    "avg_metrics": {
      "METEOR": 0.3521,
      "ROUGE_L": 0.4123,
      "CIDEr": 1.2345,
      "SPICE": 0.2134
    }
  },
  "samples": [...]
}
```

**评测指标：**
| 指标 | 说明 |
|------|------|
| METEOR | 考虑同义词和词形变化的匹配 |
| ROUGE-L | 最长公共子序列 |
| CIDEr-D | 基于 TF-IDF 的共识度量 |
| SPICE | 基于场景图的语义匹配 (需 Java) |

---

## 🔮 推理使用

### 方式一：Python 代码

```python
from inference.infer import ImageCaptioner, load_captioner

# 方法1: 使用便捷函数一键加载
captioner = load_captioner(checkpoint="outputs/ckpts/epoch19.pth")

# 方法2: 手动加载模型
from models.vit_encoder_decoder import ImageCaptionModel
import torch

model = ImageCaptionModel(vocab_size=109)
model.load_state_dict(torch.load("outputs/ckpts/epoch19.pth"))
captioner = ImageCaptioner(model)

# 单张图片推理
caption = captioner.predict("path/to/image.jpg")
print(caption)  # "the sweater this lady wears has long sleeves ..."

# 返回 ID 序列
result = captioner.predict("image.jpg", return_ids=True)
print(result)  # {"text": "...", "ids": [3, 35, 30, ...]}

# 批量推理
captions = captioner.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

# 文件夹推理
results = captioner.predict_folder("path/to/folder", output_json="results.json")
```

### 方式二：命令行

```bash
# 单张图片
python -m inference.infer \
    --checkpoint outputs/ckpts/epoch19.pth \
    --image path/to/image.jpg

# 批量推理文件夹
python -m inference.infer \
    --checkpoint outputs/ckpts/epoch19.pth \
    --folder path/to/images \
    --output results.json
```

---

## 🔧 扩展开发

如果你要添加新的训练任务：

1. 在 `trains/` 下创建新的训练脚本 (如 `task7.py`)
2. 训练完成后得到权重文件
3. 使用评测模块进行统一评测：

```python
from eval.evaluate import run_full_evaluation

results = run_full_evaluation(
    model=your_model,
    data_json="data/test.json",
    image_root="/your/image/path",
    experiment_name="task7_experiment",
)
```

---

## 📝 License

MIT License


















