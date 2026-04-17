#!/bin/bash
set -e
#
# Hyperparameter sweep for TVD on TOFU with microsoft/phi-1_5.
# Runs every combination of learning rate and lambda values defined below,
# one at a time on a single GPU. Each run gets a unique task_name so results
# never overwrite each other.
#
# Usage:
#   bash scripts/tofu_tvd_sweep_phi-1_5.sh
#
# To restrict to a single forget split, set FORGET_SPLIT before running:
#   FORGET_SPLIT=forget10 bash scripts/tofu_tvd_sweep_phi-1_5.sh
#
# To disable bfloat16 (use model default dtype instead):
#   DTYPE=default bash scripts/tofu_tvd_sweep_phi-1_5.sh

model="phi-1_5"
base_model="microsoft/phi-1_5"
model_path="saves/finetune/tofu_${model}_full"

per_device_train_batch_size=4
gradient_accumulation_steps=8   # effective batch size 32 on single GPU
num_train_epochs=20

# Set DTYPE=default to use the model's default dtype instead of bfloat16.
if [ "${DTYPE:-bfloat16}" = "default" ]; then
    dtype_arg=""
else
    dtype_arg="++model.model_args.torch_dtype=bfloat16"
fi

########################################################################################################################
# Hyperparameter grid — edit these arrays to add/remove values
########################################################################################################################

learning_rates=(
    "1e-5"
    "5e-5"
    "1e-4"
)

# Each entry is "lambda_reconstruct lambda_data lambda_orth lambda_norm"
lambda_combos=(
    "1.0  1.0  1.0  0.01"   # baseline
    "0.5  1.0  1.0  0.01"   # softer reconstruct constraint
    "1.0  1.0  5.0  0.01"   # stronger orth push
    "0.5  1.0  5.0  0.01"   # softer reconstruct + stronger orth
    "1.0  2.0  1.0  0.01"   # stronger data term
    "1.0  2.0  5.0  0.01"   # stronger data + stronger orth
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

total=$(( ${#learning_rates[@]} * ${#lambda_combos[@]} * ${#splits[@]} ))
run=0

for split in "${splits[@]}"; do
    forget_split=$(echo $split | awk '{print $1}')
    holdout_split=$(echo $split | awk '{print $2}')
    retain_split=$(echo $split | awk '{print $3}')

    retain_logs_json="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
    if [ -f "$retain_logs_json" ]; then
        retain_logs_arg="retain_logs_path=${retain_logs_json}"
    else
        echo "Warning: ${retain_logs_json} not found — forget_quality/privleak metrics will be skipped."
        retain_logs_arg="retain_logs_path=null"
    fi

    for lr in "${learning_rates[@]}"; do
        for combo in "${lambda_combos[@]}"; do
            lr_clean=$(echo $lr | tr -d ' ')
            r=$(echo $combo | awk '{print $1}')
            d=$(echo $combo | awk '{print $2}')
            o=$(echo $combo | awk '{print $3}')
            n=$(echo $combo | awk '{print $4}')

            task_name=tofu_${model}_${forget_split}_TVD_lr${lr_clean}_r${r}_d${d}_o${o}_n${n}
            run=$(( run + 1 ))
            echo "[$run/$total] ${task_name}"

            # Unlearn
            CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/tofu/default.yaml \
                trainer=TVD \
                task_name=${task_name} \
                model=${model} \
                forget_split=${forget_split} \
                retain_split=${retain_split} \
                model.model_args.pretrained_model_name_or_path=${model_path} \
                ${dtype_arg} \
                ${retain_logs_arg} \
                trainer.args.learning_rate=${lr_clean} \
                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                trainer.args.num_train_epochs=${num_train_epochs} \
                trainer.args.gradient_checkpointing=false \
                trainer.method_args.base_model_name_or_path=${base_model} \
                trainer.method_args.lambda_reconstruct=${r} \
                trainer.method_args.lambda_data=${d} \
                trainer.method_args.lambda_orth=${o} \
                trainer.method_args.lambda_norm=${n}

            # Eval
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                model=${model} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                paths.output_dir=saves/unlearn/${task_name}/evals \
                ${retain_logs_arg}
        done
    done
done

echo "Sweep complete: $total runs finished."
