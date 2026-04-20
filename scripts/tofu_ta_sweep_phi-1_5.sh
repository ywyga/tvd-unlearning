#!/bin/bash
set -e
#
# Hyperparameter sweep for TaskArithmetic on TOFU with microsoft/phi-1_5.
# Runs every combination of scale values and forget splits defined below.
# Each run gets a unique task_name so results never overwrite each other.
#
# Prerequisites:
#   The forget-only finetune checkpoints must exist before running this sweep:
#     bash scripts/tofu_finetune_phi-1_5.sh
#   This produces saves/finetune/tofu_phi-1_5_{forget_split}/ for each split.
#
# Usage:
#   bash scripts/tofu_ta_sweep_phi-1_5.sh
#
# To restrict to a single forget split:
#   FORGET_SPLIT=forget10 bash scripts/tofu_ta_sweep_phi-1_5.sh
#
# To disable bfloat16 (use model default dtype instead):
#   DTYPE=default bash scripts/tofu_ta_sweep_phi-1_5.sh
#
# To group results under a subfolder:
#   GROUP=sweep_ta_v1 bash scripts/tofu_ta_sweep_phi-1_5.sh
#   → saves to saves/unlearn/sweep_ta_v1/<task_name>/

model="phi-1_5"
base_model="microsoft/phi-1_5"
model_path="saves/finetune/tofu_${model}_full"

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

# Forget splits to evaluate. Override with FORGET_SPLIT env var to run one.
if [ -n "$FORGET_SPLIT" ]; then
    case "$FORGET_SPLIT" in
        forget01) splits=("forget01 holdout01 retain99") ;;
        forget05) splits=("forget05 holdout05 retain95") ;;
        forget10) splits=("forget10 holdout10 retain90") ;;
        *) echo "Unknown FORGET_SPLIT: $FORGET_SPLIT"; exit 1 ;;
    esac
else
    splits=(
        "forget01 holdout01 retain99"
        "forget05 holdout05 retain95"
        "forget10 holdout10 retain90"
    )
fi

########################################################################################################################
# Sweep
########################################################################################################################

total=$(( ${#scales[@]} * ${#splits[@]} ))
run=0

for split in "${splits[@]}"; do
    forget_split=$(echo $split | awk '{print $1}')
    holdout_split=$(echo $split | awk '{print $2}')
    retain_split=$(echo $split | awk '{print $3}')

    forget_model_path="saves/finetune/tofu_${model}_${forget_split}"
    if [ ! -d "$forget_model_path" ]; then
        echo "Error: forget model not found at ${forget_model_path}"
        echo "Run scripts/tofu_finetune_phi-1_5.sh first to produce the forget-only checkpoint."
        exit 1
    fi

    retain_logs_json="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
    if [ -f "$retain_logs_json" ]; then
        retain_logs_arg="retain_logs_path=${retain_logs_json}"
    else
        echo "Warning: ${retain_logs_json} not found — forget_quality/privleak metrics will be skipped."
        retain_logs_arg="retain_logs_path=null"
    fi

    for scale in "${scales[@]}"; do
        task_name=tofu_${model}_${forget_split}_TaskArithmetic_scale${scale}
        run=$(( run + 1 ))
        echo "[$run/$total] ${task_name}"

        # Unlearn (closed-form — no training loop)
        CUDA_VISIBLE_DEVICES=${gpu} python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default.yaml \
            trainer=TaskArithmetic \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            paths.output_dir=${unlearn_dir}/${task_name} \
            ${dtype_arg} \
            ${retain_logs_arg} \
            trainer.method_args.base_model_name_or_path=${base_model} \
            trainer.method_args.forget_model_path=${forget_model_path} \
            trainer.method_args.scale=${scale}

        # Eval
        CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${unlearn_dir}/${task_name} \
            paths.output_dir=${unlearn_dir}/${task_name}/evals \
            ${retain_logs_arg}
    done
done

echo "Sweep complete: $total runs finished."
