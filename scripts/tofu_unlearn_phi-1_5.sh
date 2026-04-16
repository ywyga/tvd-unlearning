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

per_device_train_batch_size=4
gradient_accumulation_steps=8   # effective batch size 32 on single GPU

# TVD loss weights (override via env vars or edit directly)
LAMBDA_RECONSTRUCT=${LAMBDA_RECONSTRUCT:-1.0}
LAMBDA_DATA=${LAMBDA_DATA:-1.0}
LAMBDA_ORTH=${LAMBDA_ORTH:-1.0}
LAMBDA_NORM=${LAMBDA_NORM:-1.0}


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
            task_name=tofu_${model}_${forget_split}_TVD_r${LAMBDA_RECONSTRUCT}_d${LAMBDA_DATA}_o${LAMBDA_ORTH}_n${LAMBDA_NORM}
            extra_args="trainer.method_args.base_model_name_or_path=${base_model} \
                trainer.method_args.lambda_reconstruct=${LAMBDA_RECONSTRUCT} \
                trainer.method_args.lambda_data=${LAMBDA_DATA} \
                trainer.method_args.lambda_orth=${LAMBDA_ORTH} \
                trainer.method_args.lambda_norm=${LAMBDA_NORM} \
                trainer.args.gradient_checkpointing=false"
        else
            task_name=tofu_${model}_${forget_split}_${trainer}
            extra_args="trainer.args.gradient_checkpointing=true"
        fi

        echo "${task_name}: Unlearning ${model_path} using ${trainer}"

        # Unlearn
        CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            ++model.model_args.torch_dtype=bfloat16 \
            ${retain_logs_arg} \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            ${extra_args}

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
