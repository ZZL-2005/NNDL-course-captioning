#!/usr/bin/env bash
set -e

# 路径配置
HWDIR="/home/chenzhican/zhangzilu/hwnndl"
NNDIR="/home/chenzhican/zhangzilu/NNDL-course-captioning"

CKPT_DIR="${HWDIR}/outputs/ckpts"
TRAIN_JSON="${NNDIR}/data/train.json"
IMAGE_ROOT="/data/zilu/images"
VOCAB_JSON="${NNDIR}/data/vocab.json"

OUT_ROOT="${HWDIR}/outputs/stage1_train"
mkdir -p "${OUT_ROOT}"

cd "${NNDIR}"

run_three_epochs() {
    GPU_ID=$1
    E1=$2
    E2=$3
    E3=$4   # 可为空，用于最后一个 GPU

    for epoch in $E1 $E2 $E3; do
        # 空位跳过（例如 GPU7 第 3 个 epoch）
        if [ -z "$epoch" ]; then
            continue
        fi

        CKPT="${CKPT_DIR}/epoch${epoch}.pth"
        if [ ! -f "${CKPT}" ]; then
            echo "[WARN] checkpoint not found: ${CKPT}, skip."
            continue
        fi

        echo "===== GPU ${GPU_ID}: Epoch ${epoch} ====="
        CUDA_VISIBLE_DEVICES=${GPU_ID} \
        python eval/stage1_predict.py \
            --checkpoint "${CKPT}" \
            --data_json "${TRAIN_JSON}" \
            --image_root "${IMAGE_ROOT}" \
            --vocab_json "${VOCAB_JSON}" \
            --output_json "${OUT_ROOT}/epoch${epoch}_train_stage1.json" \
            --batch_size 128 \
            --device "cuda" &
    done
}

# 七张 GPU（跳过 GPU1）
run_three_epochs 0 0 1 2
run_three_epochs 2 3 4 5
run_three_epochs 3 6 7 8
run_three_epochs 4 9 10 11
run_three_epochs 5 12 13 14
run_three_epochs 6 15 16 17
run_three_epochs 7 18 19 ""   # 第三个空位

wait
echo "All jobs (7 GPUs, 3 epochs per GPU) finished."
