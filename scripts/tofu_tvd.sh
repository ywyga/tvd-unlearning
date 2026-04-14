#!/bin/bash
#
# Runs TVD (Task Vector Decomposition) unlearning and TOFU evaluation
# across multiple models and forget splits.
#
# TVD requires a frozen copy of M0 (base model), a trainable M_retain,
# and a trainable M_forget — approximately 3x the GPU memory of a
# single-model method. Adjust batch size / accumulation steps accordingly.
#
# Usage:
#   bash scripts/tofu_tvd.sh
#
# Override hyperparameters at the bottom of this file or via env vars:
#   LAMBDA_RECONSTRUCT=1.0 LAMBDA_DATA=1.0 bash scripts/tofu_tvd.sh

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Models to evaluate (M1 checkpoints from HuggingFace)
models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)

# Base model paths (M0) corresponding to each entry in `models`.
# Order must match the models array above.
base_models=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Forget / holdout / retain splits to evaluate
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# TVD loss weights (override via env vars or edit directly)
LAMBDA_RECONSTRUCT=${LAMBDA_RECONSTRUCT:-1.0}
LAMBDA_DATA=${LAMBDA_DATA:-1.0}
LAMBDA_ORTH=${LAMBDA_ORTH:-1.0}
LAMBDA_NORM=${LAMBDA_NORM:-1.0}

# Reduced batch size relative to single-model methods to account for
# the extra GPU memory consumed by M_forget and the frozen M0.
per_device_train_batch_size=2
gradient_accumulation_steps=8   # keeps effective batch size at 32 on 2 GPUs


########################################################################################################################
########################################### Unlearn + Eval TOFU models #################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for i in "${!models[@]}"; do
        model="${models[$i]}"
        base_model="${base_models[$i]}"

        task_name=tofu_${model}_${forget_split}_TVD
        model_path=open-unlearning/tofu_${model}_full
        echo "${task_name}: Unlearning ${model_path} using TVD (M0=${base_model})"

        # --- Unlearn ---
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer=TVD \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.method_args.base_model_name_or_path=${base_model} \
            trainer.method_args.lambda_reconstruct=${LAMBDA_RECONSTRUCT} \
            trainer.method_args.lambda_data=${LAMBDA_DATA} \
            trainer.method_args.lambda_orth=${LAMBDA_ORTH} \
            trainer.method_args.lambda_norm=${LAMBDA_NORM} \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=false

        # --- Eval ---
        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${task_name}/evals \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
    done
done
