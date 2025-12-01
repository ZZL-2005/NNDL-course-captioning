# python scripts/eval.py --json /home/chenzhican/zhangzilu/hwnndl/outputs/test_results/epoch1_tokens.json
import json
import argparse
from tqdm import tqdm
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def evaluate(data, id2token):
    """
    data: epochX_test.json 加载后的列表
    id2token: 将 id 序列还原为 token
    """

    gts = {}   # ground truth: {id: ["sentence"]}
    res = {}   # prediction : {id: ["sentence"]}

    for i, item in enumerate(data):
        gt_ids = item["gt_ids"]
        pred_ids = item["pred_ids"]

        # id → token → sentence
        gt_tokens = [id2token[str(t)] for t in gt_ids]
        pred_tokens = [id2token[str(t)] for t in pred_ids]

        gt_sentence = " ".join(gt_tokens)
        pred_sentence = " ".join(pred_tokens)

        gts[i] = [gt_sentence]
        res[i] = [pred_sentence]

    # --- METEOR ---
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(gts, res)
    
    # --- ROUGE-L ---
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)

    # --- CIDEr-D ---
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    # --- SPICE ---
    try:
        spice_scorer = Spice()
        spice_score, _ = spice_scorer.compute_score(gts, res)
    except Exception as e:
        print("[Warning] SPICE evaluation failed (Java missing?). Error:", e)
        spice_score = -1

    return {
        "METEOR": meteor_score,
        "ROUGE_L": rouge_score,
        "CIDEr_D": cider_score,
        "SPICE": spice_score
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="the ground truth Vs prediction json file")
    parser.add_argument("--vocab", type=str, default="processed/vocab.json")
    args = parser.parse_args()

    print("[INFO] Loading test file:", args.json)
    data = load_json(args.json)
    vocab = json.load(open(args.vocab))
    id2token = vocab["id2token"]

    print("[INFO] Evaluating...")
    scores = evaluate(data, id2token)

    print("\n======= Evaluation Results =======")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
    print("=================================\n")

if __name__ == "__main__":
    main()
