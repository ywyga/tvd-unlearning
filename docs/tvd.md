# Task Vector Decomposition (TVD)

TVD is an unlearning method that decomposes the full *task vector* of a fine-tuned model into orthogonal retain and forget components, then discards the forget component to produce the unlearned model.

---

## Method

**Setup.** Let:
- **M0** — the frozen base (pre-trained) model
- **M1** — the target model, fine-tuned on the full dataset (retain + forget)
- **TV1 = M1 − M0** — the full task vector

**Training.** Starting from M1, two models are jointly trained:
- **M_retain** — optimised on the retain split → `TV_retain = M_retain − M0`
- **M_forget** — optimised on the forget split → `TV_forget = M_forget − M0`

Both models share a single joint optimizer.

**Loss function.**

```
L_total = λ_data        · L_data
        + λ_reconstruct · L_reconstruct
        + λ_orth        · L_orth
        + λ_norm        · L_norm
```

| Term | Formula | Purpose |
|------|---------|---------|
| `L_data` | `NLL(M_retain, D_retain) + NLL(M_forget, D_forget)` | Each model fits its corresponding data split |
| `L_reconstruct` | `‖TV_retain + TV_forget − TV1‖²` | Decomposition is faithful: retain + forget ≈ original |
| `L_orth` | `cos_sim(TV_retain, TV_forget)²` | Retain and forget directions are orthogonal |
| `L_norm` | `(‖TV_retain‖ / ‖TV1‖ − 1)²` | Retain vector has the same scale as the original task vector |

**Output model.** `M0 + TV_retain` — which is simply `self.model` (M_retain) after training, as it already encodes `M0 + TV_retain` by construction.

---

## Implementation

| File | Role |
|------|------|
| [`src/trainer/unlearn/tvd.py`](../src/trainer/unlearn/tvd.py) | `TVD` trainer class |
| [`configs/trainer/TVD.yaml`](../configs/trainer/TVD.yaml) | Hydra config with default hyperparameters |
| [`scripts/tofu_tvd.sh`](../scripts/tofu_tvd.sh) | Baseline script for TOFU unlearning + evaluation |

### Key design points

- **Task vectors on CPU.** `TV1` is computed once in `__init__` and stored as a list of CPU tensors to avoid occupying a fourth copy of the model on GPU.
- **Joint optimizer.** `create_optimizer` is overridden to add M_forget's parameters as a second param group to the standard HF optimizer — no separate optimizer or callback is needed.
- **Scale-invariant norm loss.** `L_norm` is expressed as `(‖TV_retain‖ / ‖TV1‖ − 1)²` so the penalty is dimensionless and the default weight `λ_norm = 1.0` works across model sizes.
- **Per-step TensorBoard logging.** `compute_loss` calls `self.log({...})` with all four loss components, following the same pattern as [`PDU`](../src/trainer/unlearn/pdu.py).

---

## Configuration

```yaml
# configs/trainer/TVD.yaml
defaults:
  - finetune

handler: TVD
method_args:
  base_model_name_or_path: ???   # required: HF Hub ID or local path to M0
  lambda_reconstruct: 1.0
  lambda_data: 1.0
  lambda_orth: 1.0
  lambda_norm: 1.0
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model_name_or_path` | **required** | HuggingFace ID or local path to M0. Must be the same architecture as the fine-tuned model. |
| `lambda_reconstruct` | `1.0` | Weight for `L_reconstruct`. Higher values enforce a tighter decomposition. |
| `lambda_data` | `1.0` | Weight for `L_data`. Controls how strongly each model fits its data split. |
| `lambda_orth` | `1.0` | Weight for `L_orth`. Higher values push retain/forget directions further apart. |
| `lambda_norm` | `1.0` | Weight for `L_norm`. Prevents the retain vector from collapsing or exploding. |

---

## Usage

### Single-run command

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=TVD \
  trainer.method_args.base_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct" \
  task_name=TVD_tofu_forget10
```

The unlearned model is saved to `saves/unlearn/TVD_tofu_forget10/`.

### Evaluation

```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/TVD_tofu_forget10 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  task_name=TVD_tofu_forget10_eval
```

### Baseline script (all models × all splits)

```bash
bash scripts/tofu_tvd.sh
```

Override loss weights without editing the script:

```bash
LAMBDA_RECONSTRUCT=2.0 LAMBDA_ORTH=0.5 bash scripts/tofu_tvd.sh
```

### TensorBoard

```bash
tensorboard --logdir saves/unlearn/TVD_tofu_forget10/logs
```

Logged metrics: `loss`, `loss_data`, `loss_reconstruct`, `loss_orth`, `loss_norm`.

---

## Memory requirements

TVD holds three full model copies in GPU memory simultaneously:

| Copy | Role | Trainable |
|------|------|-----------|
| M0 (`ref_model`) | Frozen reference for task vector computation | No |
| M_retain (`model`) | Final unlearned model | Yes |
| M_forget (`forget_model`) | Captures forget-split knowledge | Yes |

Additionally, `TV1` (one model's worth of parameters) is stored on CPU.

**Recommended batch sizes for TOFU** (adjust `gradient_accumulation_steps` to keep effective batch size at 32):

| Model | GPU VRAM | `per_device_train_batch_size` |
|-------|----------|-------------------------------|
| Llama-3.2-1B | 24 GB | 4 |
| Llama-3.2-3B | 40 GB | 2–4 |
| Llama-3.1-8B | 80 GB | 2 |

`gradient_checkpointing` is **not compatible** with TVD because activations from both M_retain and M_forget must be held in memory simultaneously for the joint backward pass. Keep `gradient_checkpointing=false` (the default).
