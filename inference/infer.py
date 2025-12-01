"""
推理模块
========
提供便捷的推理接口，支持：
    - 单张图片推理
    - 批量图片推理
    - 从文件夹推理

使用示例:
---------
    from inference.infer import ImageCaptioner
    from models.vit_encoder_decoder import ImageCaptionModel
    import torch

    # 加载模型
    model = ImageCaptionModel(vocab_size=109)
    model.load_state_dict(torch.load("outputs/ckpts/epoch19.pth"))

    # 创建推理器
    captioner = ImageCaptioner(model, vocab_json="data/vocab.json")

    # 单张图片推理
    caption = captioner.predict("path/to/image.jpg")
    print(caption)  # "the sweater this lady wears ..."

    # 批量推理
    results = captioner.predict_batch(["img1.jpg", "img2.jpg"])
"""

import os
import sys
import json
from typing import Union, List, Optional

import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class ImageCaptioner:
    """
    图像描述生成推理器
    
    Args:
        model: 加载好权重的模型
        vocab_json: 词表 JSON 文件路径
        device: 推理设备 ("cuda" / "cpu" / None 自动选择)
        transform: 图像预处理 (None 则使用默认)
    """
    
    def __init__(
        self,
        model,
        vocab_json: str = "data/vocab.json",
        device: Optional[str] = None,
        transform=None,
    ):
        # 设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 模型
        self.model = model.to(self.device)
        self.model.eval()
        
        # 词表
        with open(vocab_json, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.id2token = vocab["id2token"]
        self.token2id = vocab["token2id"]
        
        # 特殊 token
        self.pad_idx = self.token2id.get("<PAD>", 0)
        self.start_idx = self.token2id.get("<START>", 1)
        self.end_idx = self.token2id.get("<END>", 2)
        
        # 图像预处理
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def _ids_to_text(self, ids: List[int]) -> str:
        """将 token ID 序列转换为文本"""
        tokens = []
        for t in ids:
            if t == self.start_idx:
                continue
            if t == self.end_idx:
                break
            if t == self.pad_idx:
                continue
            tokens.append(self.id2token.get(str(t), f"<UNK:{t}>"))
        return " ".join(tokens)
    
    def _load_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """加载并预处理图像"""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
        
        return self.transform(img)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Image.Image],
        return_ids: bool = False,
    ) -> Union[str, dict]:
        """
        单张图片推理
        
        Args:
            image: 图片路径或 PIL Image 对象
            return_ids: 是否同时返回 token ID 序列
            
        Returns:
            如果 return_ids=False: 返回生成的文本 (str)
            如果 return_ids=True: 返回 {"text": str, "ids": list}
        """
        # 预处理
        img_tensor = self._load_image(image).unsqueeze(0).to(self.device)
        
        # 推理
        pred_ids = self.model.greedy_decode(img_tensor)  # (1, max_len)
        pred_ids = pred_ids[0].tolist()
        
        # 转文本
        text = self._ids_to_text(pred_ids)
        
        # 清理 ID 序列 (去除特殊 token)
        clean_ids = []
        for t in pred_ids:
            if t == self.start_idx:
                continue
            if t == self.end_idx:
                break
            if t == self.pad_idx:
                continue
            clean_ids.append(t)
        
        if return_ids:
            return {"text": text, "ids": clean_ids}
        return text
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Image.Image]],
        return_ids: bool = False,
    ) -> List[Union[str, dict]]:
        """
        批量图片推理
        
        Args:
            images: 图片路径列表或 PIL Image 列表
            return_ids: 是否同时返回 token ID 序列
            
        Returns:
            结果列表
        """
        if len(images) == 0:
            return []
        
        # 预处理所有图片
        img_tensors = torch.stack([self._load_image(img) for img in images])
        img_tensors = img_tensors.to(self.device)
        
        # 批量推理
        pred_ids = self.model.greedy_decode(img_tensors)  # (B, max_len)
        pred_ids = pred_ids.tolist()
        
        # 转换结果
        results = []
        for ids in pred_ids:
            text = self._ids_to_text(ids)
            
            if return_ids:
                clean_ids = []
                for t in ids:
                    if t == self.start_idx:
                        continue
                    if t == self.end_idx:
                        break
                    if t == self.pad_idx:
                        continue
                    clean_ids.append(t)
                results.append({"text": text, "ids": clean_ids})
            else:
                results.append(text)
        
        return results
    
    def predict_folder(
        self,
        folder: str,
        output_json: Optional[str] = None,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> List[dict]:
        """
        对文件夹中的所有图片进行推理
        
        Args:
            folder: 图片文件夹路径
            output_json: 输出 JSON 文件路径 (可选)
            extensions: 支持的图片扩展名
            
        Returns:
            结果列表 [{"img": filename, "caption": text}, ...]
        """
        # 收集所有图片
        images = []
        filenames = []
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith(extensions):
                images.append(os.path.join(folder, filename))
                filenames.append(filename)
        
        if len(images) == 0:
            print(f"[Warning] 文件夹中没有找到图片: {folder}")
            return []
        
        print(f"[INFO] 找到 {len(images)} 张图片，开始推理...")
        
        # 批量推理
        captions = self.predict_batch(images, return_ids=False)
        
        # 整理结果
        results = []
        for filename, caption in zip(filenames, captions):
            results.append({
                "img": filename,
                "caption": caption
            })
        
        # 保存到文件
        if output_json:
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] 结果已保存到: {output_json}")
        
        return results


# ============ 便捷函数 ============

def load_captioner(
    checkpoint: str,
    vocab_json: str = "data/vocab.json",
    device: Optional[str] = None,
    **model_kwargs,
) -> ImageCaptioner:
    """
    便捷函数：加载模型并创建推理器
    
    Args:
        checkpoint: 模型权重路径
        vocab_json: 词表 JSON 路径
        device: 设备
        **model_kwargs: 模型参数 (vocab_size, d_model, n_heads, num_layers, max_len)
        
    Returns:
        ImageCaptioner 实例
    """
    from models.vit_encoder_decoder import ImageCaptionModel
    
    # 默认模型参数
    default_kwargs = {
        "vocab_size": 109,
        "d_model": 512,
        "n_heads": 8,
        "num_layers": 4,
        "max_len": 128,
    }
    default_kwargs.update(model_kwargs)
    
    # 创建模型
    model = ImageCaptionModel(**default_kwargs)
    
    # 加载权重
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[INFO] 加载权重: {checkpoint}")
    
    return ImageCaptioner(model, vocab_json=vocab_json, device=device)


# ============ 命令行接口 ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="图像描述生成推理")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--image", type=str, default=None, help="单张图片路径")
    parser.add_argument("--folder", type=str, default=None, help="图片文件夹路径")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("--vocab_json", type=str, default="data/vocab.json", help="词表路径")
    parser.add_argument("--device", type=str, default=None, help="设备")
    
    args = parser.parse_args()
    
    if args.image is None and args.folder is None:
        parser.error("请指定 --image 或 --folder")
    
    # 加载推理器
    captioner = load_captioner(
        checkpoint=args.checkpoint,
        vocab_json=args.vocab_json,
        device=args.device,
    )
    
    # 推理
    if args.image:
        result = captioner.predict(args.image, return_ids=True)
        print(f"\n图片: {args.image}")
        print(f"描述: {result['text']}")
        print(f"IDs:  {result['ids']}")
    
    if args.folder:
        results = captioner.predict_folder(
            folder=args.folder,
            output_json=args.output,
        )
        print(f"\n共处理 {len(results)} 张图片")
        for r in results[:5]:  # 只打印前5个
            print(f"  {r['img']}: {r['caption']}")
        if len(results) > 5:
            print(f"  ... (共 {len(results)} 张)")
