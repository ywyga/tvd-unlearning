#!/bin/bash
set -e
#
# Runs unlearning + TOFU evaluation for microsoft/phi-1_5 on a single GPU.
# Assumes retain and full finetune checkpoints already exist (run
# scripts/tofu_finetune_phi-1_5.sh first, or point model_path at a HuggingFace
# hub checkpoint).
#
# Usage:
#   bash scripts/tofu_unlearn_phi-1_5.sh
#
# TVD-specific lambda overrides:
#   LAMBDA_RECONSTRUCT=1.0 LAMBDA_DATA=1.0 bash scripts/tofu_unlearn_phi-1_5.sh
#
# To disable bfloat16 (use model default dtype instead):
#   DTYPE=default bash scripts/tofu_unlearn_phi-1_5.sh
#
# To group results under a subfolder:
#   GROUP=experiment_v1 bash scripts/tofu_unlearn_phi-1_5.sh
#   → saves to saves/unlearn/experiment_v1/<task_name>/

model="phi-1_5"
base_model="microsoft/phi-1_5"   # frozen M0 for TVD

trainers_experiments=(
    "TVD        unlearn/tofu/default.yaml"
    "GradAscent unlearn/tofu/default.yaml"
    "GradDiff   unlearn/tofu/default.yaml"
    "NPO        unlearn/tofu/default.yaml"
    "DPO        unlearn/tofu/idk.yaml"
    "RMU        unlearn/tofu/default.yaml"
)

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

gpu=0
per_device_train_batch_size=4
gradient_accumulation_steps=8   # effective batch size 32 on single GPU

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

LEARNING_RATE=${LEARNING_RATE:-1e-5}

# TVD loss weights (override via env vars or edit directly)
LAMBDA_RECONSTRUCT=${LAMBDA_RECONSTRUCT:-1.0}
LAMBDA_DATA=${LAMBDA_DATA:-1.0}
LAMBDA_ORTH=${LAMBDA_ORTH:-1.0}
LAMBDA_NORM=${LAMBDA_NORM:-0.01}


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    # Only pass retain_logs_path when the file exists. Passing a non-null path
    # that doesn't exist causes a hard crash in the evaluator. When null,
    # forget_quality and privleak metrics are skipped gracefully.
    retain_logs_json="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
    if [ -f "$retain_logs_json" ]; then
        retain_logs_arg="retain_logs_path=${retain_logs_json}"
    else
        echo "Warning: ${retain_logs_json} not found — forget_quality/privleak metrics will be skipped."
        retain_logs_arg="retain_logs_path=null"
    fi

    for trainer_experiment in "${trainers_experiments[@]}"; do
        trainer=$(echo $trainer_experiment | awk '{print $1}')
        experiment=$(echo $trainer_experiment | awk '{print $2}')

        model_path=saves/finetune/tofu_${model}_full

        # Build trainer-specific extra args; for TVD encode lambdas in task_name
        # so runs with different hyperparameters don't overwrite each other.
        extra_args=""
        if [ "$trainer" = "TVD" ]; then
            task_name=tofu_${model}_${forget_split}_TVD_lr${LEARNING_RATE}_r${LAMBDA_RECONSTRUCT}_d${LAMBDA_DATA}_o${LAMBDA_ORTH}_n${LAMBDA_NORM}
            extra_args="trainer.method_args.base_model_name_or_path=${base_model} \
                trainer.method_args.lambda_reconstruct=${LAMBDA_RECONSTRUCT} \
                trainer.method_args.lambda_data=${LAMBDA_DATA} \
                trainer.method_args.lambda_orth=${LAMBDA_ORTH} \
                trainer.method_args.lambda_norm=${LAMBDA_NORM} \
                trainer.args.learning_rate=${LEARNING_RATE} \
                trainer.args.gradient_checkpointing=false"
        else
            task_name=tofu_${model}_${forget_split}_${trainer}_lr${LEARNING_RATE}
            extra_args="trainer.args.learning_rate=${LEARNING_RATE} \
                trainer.args.gradient_checkpointing=true"
        fi

        echo "${task_name}: Unlearning ${model_path} using ${trainer}"

        # Unlearn
        CUDA_VISIBLE_DEVICES=${gpu} python src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            paths.output_dir=${unlearn_dir}/${task_name} \
            ${dtype_arg} \
            ${retain_logs_arg} \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            ${extra_args}

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
