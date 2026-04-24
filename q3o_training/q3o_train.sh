#!/bin/bash

#SBATCH --job-name=q3o_shuffled_audio
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --output=ACL_LLM/logs/%j_%x.out
#SBATCH --error=ACL_LLM/logs/%j_%x.err

set -e

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate denv

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# =============================================================================
# CONFIGURATION
# =============================================================================

export MODEL_NAME="Qwen/Qwen3-Omni-30B-A3B-Thinking"
export DATASET_DIR="ACL_LLM/ms_swift_dataset"
export OUTPUT_DIR="ACL_LLM/swift_output/q3o_shuffled_audio"
export MASTER_PORT=$(shuf -i 30000-65000 -n 1)
export USE_AUDIO_IN_VIDEO=true

# USE HUGGINGFACE INSTEAD OF MODELSCOPE
export USE_HF=1

export CACHE_DIR="ACL_LLM/cache"
export HF_HOME="${CACHE_DIR}"
export HF_DATASETS_CACHE="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}"
export TORCH_HOME="${CACHE_DIR}/torch"

mkdir -p ${OUTPUT_DIR}

# Hardware
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1

# Video settings
export MAX_PIXELS=18816
export VIDEO_MAX_PIXELS=18816
export FPS_MAX_FRAMES=2
export FPS_MIN_FRAMES=2
export ENABLE_AUDIO_OUTPUT=False
export SWIFT_USE_DISTRIBUTED=0

# Memory
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# =============================================================================
# AUTO-RESUME FROM CHECKPOINT
# =============================================================================

RESUME_ARG=""
if [ -d "${OUTPUT_DIR}" ]; then
    LATEST_CKPT=$(find ${OUTPUT_DIR} -maxdepth 2 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
        echo "RESUMING FROM: $LATEST_CKPT"
        RESUME_ARG="--resume_from_checkpoint ${LATEST_CKPT}"
    fi
fi

# =============================================================================
# TRAINING
# =============================================================================

echo "=========================================="
echo "Starting Training"
echo "=========================================="

swift sft \
    --model ${MODEL_NAME} \
    --dataset "${DATASET_DIR}/train_qa_shuffled.jsonl" \
    --val_dataset "${DATASET_DIR}/test_qa.jsonl" \
    --output_dir ${OUTPUT_DIR} \
    ${RESUME_ARG} \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 5\
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --add_version false \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_length 1500 \
    --gradient_checkpointing true \
    --eval_steps 10000 \
    --save_steps 100 \
    --save_total_limit 50 \
    --logging_steps 10 \
    --split_dataset_ratio 0 \
    --packing false \
    --padding_free true \
    --optim adamw_torch_fused \
    --load_from_cache_file true \
    --dataloader_num_workers 2 \
    --dataset_num_proc 16

echo "=========================================="
echo "Training Complete! Resubmit to continue."
echo "=========================================="