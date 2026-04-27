#!/bin/bash
set -e
#
# Runs unlearning + Detoxify evaluation for microsoft/phi-1_5 on Civil Comments.
# Assumes finetune checkpoints already exist (run scripts/civil_comments_finetune_phi-1_5.sh first).
#
# Usage:
#   bash scripts/civil_comments_unlearn_phi-1_5.sh
#
# All gradient-based methods (TVD, GradAscent, etc.) are trained on the same
# subset of the Civil Comments corpus, controlled by MAX_SAMPLES and SUBSET_SEED:
#   MAX_SAMPLES=200   — number of documents per forget/retain split (set to "" for full corpus)
#   SUBSET_SEED=42    — shuffle seed (ensures identical subset across all methods)
#
# Subset consistency: when MAX_SAMPLES=200, all gradient-based methods receive
# data.forget.CIVIL_COMMENTS_forget.args.max_samples=200 and shuffle_seed=42.
# TaskArithmetic uses the M_forget_subset_N checkpoint (trained on those same N samples).
#
# TVD-specific lambda overrides:
#   LAMBDA_RECONSTRUCT=1.0 LAMBDA_DATA=1.0 bash scripts/civil_comments_unlearn_phi-1_5.sh
#
# To disable bfloat16 (use model default dtype instead):
#   DTYPE=default bash scripts/civil_comments_unlearn_phi-1_5.sh
#
# To group results under a subfolder:
#   GROUP=experiment_v1 bash scripts/civil_comments_unlearn_phi-1_5.sh
#   → saves to saves/unlearn/experiment_v1/<task_name>/

model="phi-1_5"
base_model="microsoft/phi-1_5"   # frozen M0 for TVD and TaskArithmetic

model_path="saves/finetune/civil_comments_${model}_full"

trainers_experiments=(
    "TaskArithmetic unlearn/civil_comments/task_arithmetic.yaml"
    "TVD            unlearn/civil_comments/tvd.yaml"
    "GradAscent     unlearn/civil_comments/default.yaml"
    "GradDiff       unlearn/civil_comments/default.yaml"
    "NPO            unlearn/civil_comments/default.yaml"
    "RMU            unlearn/civil_comments/default.yaml"
)

# TaskArithmetic scale hyperparameter (negation strength)
TA_SCALE=${TA_SCALE:-1.0}

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

# Subset training: MAX_SAMPLES documents per split, selected with SUBSET_SEED shuffle.
# Set MAX_SAMPLES="" to use the full corpus.
MAX_SAMPLES=${MAX_SAMPLES:-200}
SUBSET_SEED=${SUBSET_SEED:-42}

# Choose M_forget checkpoint based on whether subset or full corpus training.
# For TaskArithmetic to be fair, it should use the same subset as other methods.
if [ -n "${MAX_SAMPLES}" ]; then
    # Subset training: use M_forget_subset_N (trained on N samples)
    forget_model_path="saves/finetune/civil_comments_${model}_forget_subset_${MAX_SAMPLES}"
else
    # Full corpus: use M_forget_full
    forget_model_path="saves/finetune/civil_comments_${model}_forget"
fi

# Build the max_samples override block (applied to all gradient-based methods).
# Empty when MAX_SAMPLES is unset to allow full-corpus runs.
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
# Unlearn
########################################################################################################################

for trainer_experiment in "${trainers_experiments[@]}"; do
    trainer=$(echo $trainer_experiment | awk '{print $1}')
    experiment=$(echo $trainer_experiment | awk '{print $2}')

    # Build trainer-specific extra args; encode hyperparameters in task_name
    # so runs with different settings don't overwrite each other.
    extra_args=""
    if [ "$trainer" = "TaskArithmetic" ]; then
        if [ ! -d "$forget_model_path" ]; then
            echo "Error: forget model not found at ${forget_model_path}"
            echo "Run scripts/civil_comments_finetune_phi-1_5.sh first."
            exit 1
        fi
        task_name=civil_comments_${model}_TaskArithmetic_scale${TA_SCALE}${ms_suffix}
        extra_args="trainer.method_args.base_model_name_or_path=${base_model} \
            trainer.method_args.forget_model_path=${forget_model_path} \
            trainer.method_args.scale=${TA_SCALE}"
        # TaskArithmetic does not use training data — no subset_args
    elif [ "$trainer" = "TVD" ]; then
        task_name=civil_comments_${model}_TVD_lr${LEARNING_RATE}_r${LAMBDA_RECONSTRUCT}_d${LAMBDA_DATA}_o${LAMBDA_ORTH}_n${LAMBDA_NORM}${ms_suffix}
        extra_args="trainer.method_args.base_model_name_or_path=${base_model} \
            trainer.method_args.lambda_reconstruct=${LAMBDA_RECONSTRUCT} \
            trainer.method_args.lambda_data=${LAMBDA_DATA} \
            trainer.method_args.lambda_orth=${LAMBDA_ORTH} \
            trainer.method_args.lambda_norm=${LAMBDA_NORM} \
            trainer.args.learning_rate=${LEARNING_RATE} \
            trainer.args.gradient_checkpointing=false \
            ${subset_args}"
    else
        task_name=civil_comments_${model}_${trainer}_lr${LEARNING_RATE}${ms_suffix}
        extra_args="trainer.args.learning_rate=${LEARNING_RATE} \
            trainer.args.gradient_checkpointing=true \
            ${subset_args}"
    fi

    if [ ! -d "$model_path" ]; then
        echo "Error: model not found at ${model_path}"
        echo "Run scripts/civil_comments_finetune_phi-1_5.sh first."
        exit 1
    fi

    echo "${task_name}: Unlearning ${model_path} using ${trainer}"

    # Unlearn
    CUDA_VISIBLE_DEVICES=${gpu} python src/train.py --config-name=unlearn.yaml \
        experiment=${experiment} \
        trainer=${trainer} \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        paths.output_dir=${unlearn_dir}/${task_name} \
        ${dtype_arg} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        ${extra_args}

    # Eval
    CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py \
        experiment=eval/civil_comments/default.yaml \
        model=${model} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${unlearn_dir}/${task_name} \
        paths.output_dir=${unlearn_dir}/${task_name}/evals \
        ${dtype_arg}
done
