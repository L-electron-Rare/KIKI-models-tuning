#!/usr/bin/env bash
# Train missing LoRA adapters on Qwen2.5-32B-Instruct
set -euo pipefail

BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"

DOMAINS=(
    "dsp:datasets/processed/dsp_train.jsonl"
    "emc:datasets/processed/emc_train.jsonl"
    "iot:datasets/processed/iot_train.jsonl"
    "power:datasets/processed/power_train.jsonl"
)

echo "=== Training ${#DOMAINS[@]} missing LoRA adapters ==="
echo "Start: $(date)"

for entry in "${DOMAINS[@]}"; do
    IFS=: read -r domain dataset <<< "$entry"
    if [ ! -f "$dataset" ]; then
        echo "SKIP $domain — dataset not found: $dataset"
        continue
    fi
    echo ""
    echo "=== [$domain] Training start: $(date) ==="
    python scripts/train_sft.py \
        --base-model "$BASE_MODEL" \
        --dataset "$dataset" \
        --output-dir "outputs/sft-${domain}-qwen25-32b" \
        --epochs 2 --batch-size 1 --lr 1e-4 --lora-r 8 \
        --gradient-accumulation-steps 16
    echo "=== [$domain] Done: $(date) ==="
done

# Copy to loras directory
echo ""
echo "=== Copying to /home/kxkm/loras/ ==="
for entry in "${DOMAINS[@]}"; do
    IFS=: read -r domain _ <<< "$entry"
    outdir="outputs/sft-${domain}-qwen25-32b"
    if [ -d "$outdir" ]; then
        cp -r "$outdir" "/home/kxkm/loras/mascarade-${domain}"
        echo "Copied mascarade-${domain}"
    fi
done

echo "=== All done: $(date) ==="
