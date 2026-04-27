#!/bin/bash
set -e
#
# Finetunes microsoft/phi-1_5 on Civil Comments to create three prerequisite
# checkpoints required by TVD and TaskArithmetic:
#
#   saves/finetune/civil_comments_phi-1_5_forget_subset/  — M_forget_subset: 200 toxic samples (seed 42)
#   saves/finetune/civil_comments_phi-1_5_forget/         — M_forget_full: entire toxic corpus
#   saves/finetune/civil_comments_phi-1_5_full/           — M1: all train comments
#
# When doing subset training (MAX_SAMPLES=200): TaskArithmetic uses M_forget_subset
# When doing full corpus training: TaskArithmetic uses M_forget_full
# M1 is the starting point for TVD and all gradient-based unlearning methods.
#
# Usage:
#   bash scripts/civil_comments_finetune_phi-1_5.sh           # single GPU
#   NGPUS=4 bash scripts/civil_comments_finetune_phi-1_5.sh   # 4 GPUs (DDP, ~4x faster)
#
# Multi-GPU uses configs/accelerate/ddp_4gpu.yaml (plain DDP, no DeepSpeed).
# gradient_accumulation_steps is divided by NGPUS automatically so that the
# effective batch size (per_device × NGPUS × accum) stays constant at 32.
#
# To disable bfloat16 (use model default dtype instead):
#   DTYPE=default bash scripts/civil_comments_finetune_phi-1_5.sh
#
# Prerequisites:
#   python setup_data.py --civil_comments   # downloads and filters Civil Comments to data/civil_comments/

model="phi-1_5"

NGPUS=${NGPUS:-1}
per_device_train_batch_size=4
# Keep effective batch size = 32: per_device(4) × NGPUS × gradient_accumulation_steps = 32
gradient_accumulation_steps=$(( 8 / NGPUS ))
if [ "${gradient_accumulation_steps}" -lt 1 ]; then
    gradient_accumulation_steps=1
fi

# Set DTYPE=default to use the model's default dtype instead of bfloat16.
if [ "${DTYPE:-default}" = "default" ]; then
    dtype_arg=""
else
    dtype_arg="++model.model_args.torch_dtype=bfloat16"
fi

# Build the launch command: accelerate launch for multi-GPU, plain python for single GPU.
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ "${NGPUS}" -gt 1 ]; then
    echo "Multi-GPU mode: ${NGPUS} GPUs, effective batch size = ${per_device_train_batch_size} × ${NGPUS} × ${gradient_accumulation_steps} = $(( per_device_train_batch_size * NGPUS * gradient_accumulation_steps ))"
    launch="accelerate launch --config_file configs/accelerate/ddp_4gpu.yaml --num_processes ${NGPUS} --main_process_port ${MASTER_PORT}"
    ddp_arg="trainer.args.ddp_find_unused_parameters=false"
else
    echo "Single-GPU mode"
    launch="CUDA_VISIBLE_DEVICES=0 python"
    ddp_arg=""
fi


########################################################################################################################
# M_forget_subset — fine-tune on 200 toxic samples (seed 42, required by TaskArithmetic for subset experiments)
########################################################################################################################

task_name=civil_comments_${model}_forget_subset

echo "Finetuning ${model} on Civil Comments toxic corpus (200 samples, seed 42) → ${task_name}"

${launch} src/train.py experiment=finetune/civil_comments/default.yaml \
    task_name=${task_name} \
    model=${model} \
    ${dtype_arg} \
    data/datasets@data.train=CIVIL_COMMENTS_forget \
    data.train.CIVIL_COMMENTS_forget.args.max_samples=200 \
    data.train.CIVIL_COMMENTS_forget.args.shuffle_seed=42 \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    ${ddp_arg}


########################################################################################################################
# M_forget_full — fine-tune on entire toxic corpus (baseline for TaskArithmetic)
########################################################################################################################

task_name=civil_comments_${model}_forget

echo "Finetuning ${model} on Civil Comments toxic corpus (full) → ${task_name}"

${launch} src/train.py experiment=finetune/civil_comments/default.yaml \
    task_name=${task_name} \
    model=${model} \
    ${dtype_arg} \
    data/datasets@data.train=CIVIL_COMMENTS_forget \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    ${ddp_arg}


########################################################################################################################
# M1 — fine-tune on all train comments (required by TVD and as starting point for all methods)
########################################################################################################################

task_name=civil_comments_${model}_full

echo "Finetuning ${model} on Civil Comments full train corpus → ${task_name}"

${launch} src/train.py experiment=finetune/civil_comments/default.yaml \
    task_name=${task_name} \
    model=${model} \
    ${dtype_arg} \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    ${ddp_arg}

echo "Finetune complete."
echo "  M_forget_subset → saves/finetune/civil_comments_${model}_forget_subset/"
echo "  M_forget_full   → saves/finetune/civil_comments_${model}_forget/"
echo "  M1              → saves/finetune/civil_comments_${model}_full/"
