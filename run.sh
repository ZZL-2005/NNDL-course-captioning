#!/usr/bin/env bash
set -e

# 路径配置
HWDIR="/home/chenzhican/zhangzilu/hwnndl"
NNDIR="/home/chenzhican/zhangzilu/NNDL-course-captioning"

CKPT_DIR="${HWDIR}/outputs/ckpts"          # epoch0.pth ~ epoch19.pth
TEST_JSON="${NNDIR}/data/test.json"
IMAGE_ROOT="/data/zilu/images"
VOCAB_JSON="${NNDIR}/data/vocab.json"
OUT_ROOT="${NNDIR}/outputs/eval_results"   # evaluate.py 默认也是往这里写

cd "${NNDIR}"

for epoch in $(seq 0 19); do
    CKPT="${CKPT_DIR}/epoch${epoch}.pth"
    if [ ! -f "${CKPT}" ]; then
        echo "[WARN] checkpoint not found: ${CKPT}, skip."
        continue
    fi

    EXP_NAME="epoch${epoch}_test"
    echo "===== Evaluating TEST for ${EXP_NAME} ====="

    python eval/evaluate.py \
        --checkpoint "${CKPT}" \
        --data_json "${TEST_JSON}" \
        --image_root "${IMAGE_ROOT}" \
        --vocab_json "${VOCAB_JSON}" \
        --output_dir "${OUT_ROOT}" \
        --experiment_name "${EXP_NAME}" \
        --device "cuda"
done

echo "All test evaluations done. See: ${OUT_ROOT}"