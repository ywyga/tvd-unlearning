#!/bin/bash
set -e
#
# Finetunes microsoft/phi-1_5 on WMDP-cyber corpus to create two prerequisite
# checkpoints required by TVD and TaskArithmetic:
#
#   saves/finetune/wmdp_phi-1_5_cyber_full/    — M1: combined forget+retain corpus
#   saves/finetune/wmdp_phi-1_5_cyber_forget/  — M_forget: forget-only corpus
#
# M1 is the starting point for TVD unlearning.
# M_forget is used by TaskArithmetic to build the forget task vector.
#
# Usage:
#   bash scripts/wmdp_finetune_phi-1_5.sh
#
# To disable bfloat16 (use model default dtype instead):
#   DTYPE=default bash scripts/wmdp_finetune_phi-1_5.sh
#
# Prerequisites:
#   python setup_data.py --wmdp   # downloads WMDP corpora to data/wmdp/

model="phi-1_5"
data_split="cyber"

gpu=0
per_device_train_batch_size=4
gradient_accumulation_steps=8   # effective batch size 32 on single GPU

# Set DTYPE=default to use the model's default dtype instead of bfloat16.
if [ "${DTYPE:-default}" = "default" ]; then
    dtype_arg=""
else
    dtype_arg="++model.model_args.torch_dtype=bfloat16"
fi


########################################################################################################################
# M_forget — fine-tune on forget-only corpus (required by TaskArithmetic)
########################################################################################################################

task_name=wmdp_${model}_${data_split}_forget

echo "Finetuning ${model} on WMDP ${data_split}-forget corpus → ${task_name}"

CUDA_VISIBLE_DEVICES=${gpu} python src/train.py experiment=finetune/wmdp/default.yaml \
    task_name=${task_name} \
    model=${model} \
    ${dtype_arg} \
    data/datasets@data.train=WMDP_forget \
    "data.train.WMDP_forget.args.hf_args.data_files=data/wmdp/wmdp-corpora/${data_split}-forget-corpus.jsonl" \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}


########################################################################################################################
# M1 — fine-tune on combined forget+retain corpus (required by TVD and as starting point for all methods)
########################################################################################################################

task_name=wmdp_${model}_${data_split}_full

echo "Finetuning ${model} on WMDP ${data_split} combined corpus → ${task_name}"

CUDA_VISIBLE_DEVICES=${gpu} python src/train.py experiment=finetune/wmdp/default.yaml \
    task_name=${task_name} \
    model=${model} \
    ${dtype_arg} \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}

echo "Finetune complete."
echo "  M1       → saves/finetune/wmdp_${model}_${data_split}_full/"
echo "  M_forget → saves/finetune/wmdp_${model}_${data_split}_forget/"
