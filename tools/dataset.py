import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class CaptionDataset(Dataset):
    """
    JSON 格式:
    [
        {
            "img": "xxx.jpg",
            "cap_ids": [...],
            "length": 32
        }
    ]
    """
    def __init__(self, json_path, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform

        with open(json_path, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img_path_full = os.path.join(self.image_root, item["img"])
        img = Image.open(img_path_full).convert("RGB")
        if self.transform:
            img = self.transform(img)

        cap_ids = torch.tensor(item["cap_ids"], dtype=torch.long)
        length = item["length"]

        # 返回内部用的文件名（后续用于 JSON 保存）
        return img, cap_ids, length, item["img"]