#!/usr/bin/env bash
# Train all LoRA adapters on Qwen2.5-32B-Instruct (QLoRA 4-bit)
# Target: KXKM-AI RTX 4090 24GB
# Usage: bash scripts/train_all_32b.sh
set -euo pipefail

BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"
HUB_ORG="clemsail"

DOMAINS=(
    "stm32:datasets/processed/stm32_train.jsonl"
    "kicad:datasets/processed/kicad_train.jsonl"
    "embedded:datasets/processed/embedded_train.jsonl"
    "spice:datasets/processed/spice_train.jsonl"
    "platformio:datasets/processed/platformio_train.jsonl"
    "freecad:datasets/processed/freecad_train.jsonl"
    "espidf:datasets/processed/espidf_train.jsonl"
)

echo "=== Training ${#DOMAINS[@]} LoRA adapters on ${BASE_MODEL} ==="
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
        --epochs 2 \
        --batch-size 1 \
        --lr 1e-4 \
        --lora-r 8 \
        --gradient-accumulation-steps 16

    echo "=== [$domain] Done: $(date) ==="
done

echo ""
echo "=== All training complete: $(date) ==="
