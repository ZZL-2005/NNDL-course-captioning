import torch

def collate_fn(batch):
    """
    batch: list of (img, cap_ids, length, img_name)
    """
    imgs, caps, lengths, names = zip(*batch)

    imgs = torch.stack(imgs, dim=0)

    max_len = max(lengths)
    padded = torch.full((len(caps), max_len), fill_value=0, dtype=torch.long)

    for i, cap in enumerate(caps):
        padded[i, :len(cap)] = cap

    lengths = torch.tensor(lengths, dtype=torch.long)

    return imgs, padded, lengths, names



def collate_lm(batch):
    """
    batch: list of (img, cap_ids, length, img_name)
    本模型不使用 img, img_name
    """
    _, caps, lengths, _ = zip(*batch)

    max_len = max(lengths)
    padded_caps = torch.full((len(caps), max_len), 0, dtype=torch.long)

    for i, cap in enumerate(caps):
        padded_caps[i, :len(cap)] = cap

    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_caps, lengths
