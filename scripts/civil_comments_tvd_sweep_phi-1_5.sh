#!/bin/bash
set -e
#
# Hyperparameter sweep for TVD on Civil Comments with microsoft/phi-1_5.
# Runs every combination of learning rate and lambda values defined below,
# one at a time on a single GPU. Each run gets a unique task_name so results
# never overwrite each other.
#
# Prerequisites:
#   bash scripts/civil_comments_finetune_phi-1_5.sh
#   → saves/finetune/civil_comments_phi-1_5_full/  (M1)
#
# Usage:
#   bash scripts/civil_comments_tvd_sweep_phi-1_5.sh
#
# Subset training (default MAX_SAMPLES=200): all runs use the same 200-sample
# forget/retain subsets for fair comparison. Set MAX_SAMPLES="" for full corpus.
#
# To disable bfloat16 (use model default dtype instead):
#   DTYPE=default bash scripts/civil_comments_tvd_sweep_phi-1_5.sh
#
# To group results under a subfolder:
#   GROUP=sweep_tvd_v1 bash scripts/civil_comments_tvd_sweep_phi-1_5.sh
#   → saves to saves/unlearn/sweep_tvd_v1/<task_name>/

model="phi-1_5"
base_model="microsoft/phi-1_5"
model_path="saves/finetune/civil_comments_${model}_full"

gpu=0
per_device_train_batch_size=4
gradient_accumulation_steps=8   # effective batch size 32 on single GPU
num_train_epochs=5

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

# Subset training: MAX_SAMPLES documents per split, selected with SUBSET_SEED shuffle.
# Set MAX_SAMPLES="" to use the full corpus.
MAX_SAMPLES=${MAX_SAMPLES:-200}
SUBSET_SEED=${SUBSET_SEED:-42}

if [ -n "${MAX_SAMPLES}" ]; then
    subset_args="data.forget.CIVIL_COMMENTS_forget.args.max_samples=${MAX_SAMPLES} \
        data.forget.CIVIL_COMMENTS_forget.args.shuffle_seed=${SUBSET_SEED} \
        data.retain.CIVIL_COMMENTS_retain.args.max_samples=${MAX_SAMPLES} \
        data.retain.CIVIL_COMMENTS_retain.args.shuffle_seed=${SUBSET_SEED}"
    ms_suffix="_ms${MAX_SAMPLES}"
else
    subset_args=""
    ms_suffix=""
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

########################################################################################################################
# Sweep
########################################################################################################################

total=$(( ${#learning_rates[@]} * ${#lambda_combos[@]} ))
run=0

if [ ! -d "$model_path" ]; then
    echo "Error: M1 model not found at ${model_path}"
    echo "Run scripts/civil_comments_finetune_phi-1_5.sh first."
    exit 1
fi

for lr in "${learning_rates[@]}"; do
    for combo in "${lambda_combos[@]}"; do
        lr_clean=$(echo $lr | tr -d ' ')
        r=$(echo $combo | awk '{print $1}')
        d=$(echo $combo | awk '{print $2}')
        o=$(echo $combo | awk '{print $3}')
        n=$(echo $combo | awk '{print $4}')

        task_name=civil_comments_${model}_TVD_lr${lr_clean}_r${r}_d${d}_o${o}_n${n}${ms_suffix}
        run=$(( run + 1 ))
        echo "[$run/$total] ${task_name}"

        # Unlearn
        CUDA_VISIBLE_DEVICES=${gpu} python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/civil_comments/tvd.yaml \
            trainer=TVD \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            paths.output_dir=${unlearn_dir}/${task_name} \
            ${dtype_arg} \
            trainer.args.learning_rate=${lr_clean} \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.num_train_epochs=${num_train_epochs} \
            trainer.args.gradient_checkpointing=false \
            trainer.method_args.base_model_name_or_path=${base_model} \
            trainer.method_args.lambda_reconstruct=${r} \
            trainer.method_args.lambda_data=${d} \
            trainer.method_args.lambda_orth=${o} \
            trainer.method_args.lambda_norm=${n} \
            ${subset_args}

        # Eval
        CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py \
            experiment=eval/civil_comments/default.yaml \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${unlearn_dir}/${task_name} \
            paths.output_dir=${unlearn_dir}/${task_name}/evals \
            ${dtype_arg}
    done
done

echo "Sweep complete: $total runs finished."
