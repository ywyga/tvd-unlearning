#!/bin/bash
#
# Finetunes microsoft/phi-1_5 on TOFU (retain splits + full) and evaluates
# each checkpoint. Runs on a single GPU — no accelerate/DeepSpeed required.
#
# Usage:
#   bash scripts/tofu_finetune_phi-1_5.sh
#
# To disable bfloat16 (use model default dtype instead):
#   DTYPE=default bash scripts/tofu_finetune_phi-1_5.sh

model="phi-1_5"

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

per_device_train_batch_size=4
gradient_accumulation_steps=8   # effective batch size 32 on single GPU

# Set DTYPE=default to use the model's default dtype instead of bfloat16.
if [ "${DTYPE:-bfloat16}" = "default" ]; then
    dtype_arg=""
else
    dtype_arg="++model.model_args.torch_dtype=bfloat16"
fi


########################################################################################################################
########################################### RETAIN Finetuned TOFU ######################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=finetune/tofu/default.yaml \
        task_name=tofu_${model}_${retain_split} \
        model=${model} \
        ${dtype_arg} \
        data/datasets@data.train=TOFU_QA_retain \
        data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.gradient_checkpointing=true

    CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_${retain_split} \
        model=${model} \
        ${dtype_arg} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_${retain_split}
done


########################################################################################################################
########################################### FULL Finetuned TOFU ########################################################
########################################################################################################################

CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=finetune/tofu/default.yaml \
    task_name=tofu_${model}_full \
    model=${model} \
    ${dtype_arg} \
    data/datasets@data.train=TOFU_QA_full \
    data.train.TOFU_QA_full.args.hf_args.name=full \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    trainer.args.gradient_checkpointing=true

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_full_${forget_split} \
        model=${model} \
        ${dtype_arg} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        paths.output_dir=saves/eval/tofu_${model}_full/evals_${forget_split}
done
