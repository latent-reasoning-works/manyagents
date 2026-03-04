#!/bin/bash
#SBATCH --job-name=trace_extract_olmo
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/trace_extract_%j.out
#SBATCH --error=logs/trace_extract_%j.err

# ============================================================
# OLMo Reasoning Trace Extraction — Mila cluster
#
# Submit:
#   mkdir -p logs
#   sbatch scripts/slurm_extract_traces.sh
#
# Quick test (10 samples, last layer only):
#   sbatch scripts/slurm_extract_traces.sh --test
#
# Full run (500 samples, all layers):
#   sbatch scripts/slurm_extract_traces.sh
# ============================================================

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

# Environment
export HF_HOME=/network/weights/.cache/huggingface
export TRANSFORMERS_CACHE=/network/weights/.cache/huggingface

# Working directory
cd $SCRATCH/lrw/agents
source .venv/bin/activate

# Determine run mode
if [[ "${1:-}" == "--test" ]]; then
    echo "=== TEST MODE: 10 samples, last layer ==="
    N_SAMPLES=10
    LAYERS="--layers -1"
    OUTPUT_DIR="outputs/traces/olmo_gsm8k_test_$(date +%Y%m%d_%H%M%S)"
else
    echo "=== FULL MODE: 500 samples, all layers ==="
    N_SAMPLES=500
    LAYERS=""
    OUTPUT_DIR="outputs/traces/olmo_gsm8k_full_$(date +%Y%m%d_%H%M%S)"
fi

python scripts/extract_traces.py \
    --model olmo-7b \
    --dataset gsm8k \
    --n-samples "$N_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens 512 \
    --temperature 0.7 \
    $LAYERS

echo "Finished: $(date)"
echo "Output: $OUTPUT_DIR"
