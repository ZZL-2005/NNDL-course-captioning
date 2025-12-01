# 做数据的预处理：分词、构建词表、编码、划分数据集等
import os
import json
import random
import math
from typing import Dict, List, Tuple
from collections import Counter
SPECIAL_TOKENS = ["<PAD>", "<START>", "<END>"]
RANDOM_SEED = 42
CONFIG = {
        "caption_path": "/home/chenzhican/zhangzilu/NNDL-course-captioning/data/captions.json",
        "output_dir": "processed/"
    }
def tokenize(text: str) -> List[str]:
    """Lowercase + simple punctuation split."""
    text = text.lower()
    puncts = ".,?!;:\"()"
    for p in puncts:
        text = text.replace(p, f" {p} ")
    return [t.strip() for t in text.split() if t.strip()]

def build_vocab(captions: Dict[str, str], min_freq: int = 1):
    counter = Counter()
    for cap in captions.values():
        counter.update(tokenize(cap))

    vocab = {}
    id2token = {}

    idx = 0
    # add special tokens
    for t in SPECIAL_TOKENS:
        vocab[t] = idx
        id2token[idx] = t
        idx += 1

    # normal tokens
    for tok, f in counter.items():
        if f >= min_freq:
            vocab[tok] = idx
            id2token[idx] = tok
            idx += 1

    return {
        "token2id": vocab,
        "id2token": id2token,
        "freq": dict(counter)
    }

def encode_caption(tokens: List[str], vocab) -> List[int]:
    tok2id = vocab["token2id"]
    ids = [tok2id["<START>"]]
    for t in tokens:
        # 理论上所有 token 都来自同一批 captions，不会 OOV
        if t in tok2id:
            ids.append(tok2id[t])
    ids.append(tok2id["<END>"])
    return ids

def compute_prefix2next_entropy(captions: Dict[str, str]) -> float:
    """
    使用“整个前缀序列”作为条件：
        序列 t0, t1, ..., t_{n-1}
        prefix_i = (t0, ..., t_{i-1})
        next_i   = t_i

    H(Y|X) = - sum_{prefix, next} p(prefix, next) log2 p(next | prefix)
    """
    joint_counter = Counter()   # (prefix_tuple, next_token)
    prefix_counter = Counter()  # prefix_tuple

    for cap in captions.values():
        toks = tokenize(cap)
        if not toks:
            continue

        prefix: Tuple[str, ...] = ()
        for nxt in toks:
            joint_counter[(prefix, nxt)] += 1
            prefix_counter[prefix] += 1
            prefix = prefix + (nxt,)

    total_pairs = sum(joint_counter.values())
    if total_pairs == 0:
        return 0.0

    ent = 0.0
    for (prefix, nxt), c_px in joint_counter.items():
        p_px = c_px / total_pairs
        p_x_given_p = c_px / prefix_counter[prefix]
        ent -= p_px * math.log2(p_x_given_p)
    return ent

def split_keys(keys: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1):
    random.seed(RANDOM_SEED)
    random.shuffle(keys)
    n = len(keys)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return keys[:t1], keys[t1:t2], keys[t2:]

def save_dataset_json(out_path: str, keys: List[str], captions: Dict[str, str], vocab):
    """
    JSON structure:
    [
        {"img": "xxx.jpg", "cap_ids": [...], "length": X},
        ...
    ]
    """
    print(f"[Building JSON] {out_path} samples={len(keys)}")

    out_list = []
    for k in keys:
        toks = tokenize(captions[k])
        cap_ids = encode_caption(toks, vocab)

        out_list.append({
            "img": k,
            "cap_ids": cap_ids,
            "length": len(cap_ids)
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2)

    print(f"[OK] Saved {out_path}")

def main():

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ---- Load captions ----
    with open(CONFIG["caption_path"], "r", encoding="utf-8") as f:
        captions = json.load(f)

    keys = list(captions.keys())

    # ---- Vocab ----
    vocab = build_vocab(captions)
    with open(os.path.join(CONFIG["output_dir"], "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=4, ensure_ascii=False)
    print(f"[OK] Saved vocab: {len(vocab['token2id'])} tokens")

    # ---- Entropy statistics ----
    prefix_ent = compute_prefix2next_entropy(captions)
    print(f"Prefix(full)->next-token entropy H(Y|X): {prefix_ent:.4f} bits")

    # ---- Split dataset ----
    train_keys, val_keys, test_keys = split_keys(keys)
    print(f"Train: {len(train_keys)}, Val: {len(val_keys)}, Test: {len(test_keys)}")

    # ---- Save JSON ----
    save_dataset_json(os.path.join(CONFIG["output_dir"], "train.json"),
                      train_keys, captions, vocab)

    save_dataset_json(os.path.join(CONFIG["output_dir"], "val.json"),
                      val_keys, captions, vocab)

    save_dataset_json(os.path.join(CONFIG["output_dir"], "test.json"),
                      test_keys, captions, vocab)

    print("=== DONE ===")

if __name__ == "__main__":
    main()