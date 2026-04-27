#!/bin/bash
set -e
#
# Hyperparameter sweep for TaskArithmetic on Civil Comments with microsoft/phi-1_5.
# Runs every combination of scale values defined below.
# Each run gets a unique task_name so results never overwrite each other.
#
# Prerequisites:
#   The finetune checkpoints must exist before running this sweep:
#     bash scripts/civil_comments_finetune_phi-1_5.sh
#   This produces:
#     saves/finetune/civil_comments_phi-1_5_forget_subset/  (200 samples, seed 42)
#     saves/finetune/civil_comments_phi-1_5_forget/         (full toxic corpus)
#     saves/finetune/civil_comments_phi-1_5_full/           (M1)
#
# By default, this sweep uses M_forget_full (entire toxic corpus). To sweep with the
# 200-sample subset (for fair comparison with gradient methods), set SUBSET_TRAINING=1:
#   SUBSET_TRAINING=1 bash scripts/civil_comments_ta_sweep_phi-1_5.sh
#
# Usage:
#   bash scripts/civil_comments_ta_sweep_phi-1_5.sh
#
# To disable bfloat16 (use model default dtype instead):
#   DTYPE=default bash scripts/civil_comments_ta_sweep_phi-1_5.sh
#
# To group results under a subfolder:
#   GROUP=sweep_ta_v1 bash scripts/civil_comments_ta_sweep_phi-1_5.sh
#   → saves to saves/unlearn/sweep_ta_v1/<task_name>/

model="phi-1_5"
base_model="microsoft/phi-1_5"

gpu=0

# Set DTYPE=default to use the model's default dtype instead of bfloat16.
if [ "${DTYPE:-default}" = "default" ]; then
    dtype_arg=""
else
    dtype_arg="++model.model_args.torch_dtype=bfloat16"
fi

# Output directory prefix — set GROUP to nest results in a subfolder.
if [ -n "${GROUP}" ]; then
    unlearn_dir="saves/unlearn/${GROUP}"
else
    unlearn_dir="saves/unlearn"
fi

########################################################################################################################
# Hyperparameter grid — edit this array to add/remove scale values
########################################################################################################################

scales=(
    "0.1"
    "0.25"
    "0.5"
    "0.75"
    "1.0"
    "1.5"
    "2.0"
)

# Use subset training (M_forget_subset) if requested, otherwise full corpus (M_forget_full)
SUBSET_TRAINING=${SUBSET_TRAINING:-0}
if [ "$SUBSET_TRAINING" = "1" ]; then
    forget_model_path="saves/finetune/civil_comments_${model}_forget_subset"
    forget_suffix="_ms200"
else
    forget_model_path="saves/finetune/civil_comments_${model}_forget"
    forget_suffix=""
fi

model_path="saves/finetune/civil_comments_${model}_full"

########################################################################################################################
# Sweep
########################################################################################################################

total=${#scales[@]}
run=0

if [ ! -d "$forget_model_path" ]; then
    echo "Error: forget model not found at ${forget_model_path}"
    echo "Run scripts/civil_comments_finetune_phi-1_5.sh first to produce the necessary checkpoints."
    exit 1
fi

if [ ! -d "$model_path" ]; then
    echo "Error: M1 model not found at ${model_path}"
    echo "Run scripts/civil_comments_finetune_phi-1_5.sh first."
    exit 1
fi

for scale in "${scales[@]}"; do
    task_name=civil_comments_${model}_TaskArithmetic_scale${scale}${forget_suffix}
    run=$(( run + 1 ))
    echo "[$run/$total] ${task_name}"

    # Unlearn (closed-form — no training loop)
    CUDA_VISIBLE_DEVICES=${gpu} python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/civil_comments/task_arithmetic.yaml \
        trainer=TaskArithmetic \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        paths.output_dir=${unlearn_dir}/${task_name} \
        ${dtype_arg} \
        trainer.method_args.base_model_name_or_path=${base_model} \
        trainer.method_args.forget_model_path=${forget_model_path} \
        trainer.method_args.scale=${scale}

    # Eval
    CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py \
        experiment=eval/civil_comments/default.yaml \
        model=${model} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${unlearn_dir}/${task_name} \
        paths.output_dir=${unlearn_dir}/${task_name}/evals \
        ${dtype_arg}
done

echo "Sweep complete: $total runs finished."
