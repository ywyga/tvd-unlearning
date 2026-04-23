#!/bin/bash
set -e
#
# Evaluates phi-1_5 checkpoints on WMDP-cyber + MMLU using lm_eval harness.
# Runs: base model, M1 (combined finetune), and all unlearned models found
# under the unlearn directory.
#
# Usage:
#   bash scripts/wmdp_eval_phi-1_5.sh
#
# To evaluate only specific checkpoints, set MODELS as a space-separated list
# of "task_name:model_path" pairs:
#   MODELS="base:microsoft/phi-1_5 full:saves/finetune/wmdp_phi-1_5_cyber_full" \
#     bash scripts/wmdp_eval_phi-1_5.sh
#
# To group unlearn results under a subfolder:
#   GROUP=experiment_v1 bash scripts/wmdp_eval_phi-1_5.sh
#
# To disable bfloat16:
#   DTYPE=default bash scripts/wmdp_eval_phi-1_5.sh

model="phi-1_5"
base_model="microsoft/phi-1_5"
data_split="cyber"

gpu=0

if [ "${DTYPE:-default}" = "default" ]; then
    dtype_arg=""
else
    dtype_arg="++model.model_args.torch_dtype=bfloat16"
fi

if [ -n "${GROUP}" ]; then
    unlearn_dir="saves/unlearn/${GROUP}"
else
    unlearn_dir="saves/unlearn"
fi

########################################################################################################################
# Helper
########################################################################################################################

run_eval() {
    local task_name="$1"
    local model_path="$2"
    local output_dir="$3"

    echo "Evaluating: ${task_name} (${model_path})"
    CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py \
        experiment=eval/wmdp/default.yaml \
        model=${model} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        paths.output_dir=${output_dir} \
        ${dtype_arg}
}

########################################################################################################################
# Evaluate
########################################################################################################################

if [ -n "${MODELS}" ]; then
    # User-supplied list of "task_name:model_path" pairs
    for entry in ${MODELS}; do
        task_name="${entry%%:*}"
        model_path="${entry#*:}"
        run_eval "${task_name}" "${model_path}" "saves/eval/${task_name}"
    done
else
    # Default: base model + M1 finetune + all unlearned models
    run_eval \
        "wmdp_${model}_base" \
        "${base_model}" \
        "saves/eval/wmdp_${model}_base"

    full_path="saves/finetune/wmdp_${model}_${data_split}_full"
    if [ -d "${full_path}" ]; then
        run_eval \
            "wmdp_${model}_${data_split}_full" \
            "${full_path}" \
            "saves/eval/wmdp_${model}_${data_split}_full"
    else
        echo "Skipping M1 eval — ${full_path} not found"
    fi

    # All unlearned models: any subdirectory of unlearn_dir that starts with wmdp_
    if [ -d "${unlearn_dir}" ]; then
        for model_dir in "${unlearn_dir}"/wmdp_*/; do
            [ -d "${model_dir}" ] || continue
            task_name=$(basename "${model_dir}")
            run_eval \
                "${task_name}" \
                "${model_dir}" \
                "${model_dir}/evals"
        done
    else
        echo "No unlearn directory found at ${unlearn_dir} — skipping unlearned model evals"
    fi
fi

echo "Eval complete."
