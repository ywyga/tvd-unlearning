# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenUnlearning is a unified benchmarking framework for LLM machine unlearning — selectively removing knowledge from pretrained language models. It supports multiple unlearning methods, benchmarks (TOFU, MUSE, WMDP), and evaluation metrics.

## Commands

**Install**:
```bash
pip install -e .                     # Core install
pip install -e ".[lm-eval]"          # With lm-evaluation-harness support
pip install -e ".[detoxify]"         # With Detoxify toxicity evaluation support
pip install -e ".[dev]"              # With dev tools (ruff, pre-commit)
```

**Lint / Format**:
```bash
make quality   # Run ruff checks on src/, scripts/, setup.py
make style     # Auto-fix with ruff
```

**Tests**:
```bash
make test      # pytest with CUDA_VISIBLE_DEVICES= (CPU-only)
```

**Run unlearning**:
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=GradAscent \
  forget_split=forget10 \
  retain_split=retain90
```

**Run evaluation**:
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct
```

**Experiment scripts** (run multiple methods/benchmarks):
```bash
bash scripts/tofu_unlearn.sh
bash scripts/muse_unlearn.sh
bash scripts/tofu_finetune.sh

# phi-1_5 specific scripts (TOFU and WMDP-cyber)
bash scripts/tofu_finetune_phi-1_5.sh          # prereqs: full/retain/forget checkpoints
bash scripts/tofu_unlearn_phi-1_5.sh           # all methods on TOFU
bash scripts/tofu_ta_sweep_phi-1_5.sh          # TaskArithmetic scale sweep (TOFU)
bash scripts/wmdp_finetune_phi-1_5.sh              # prereqs: M1 + M_forget checkpoints
bash scripts/wmdp_unlearn_phi-1_5.sh               # all methods on WMDP-cyber (subset training)
bash scripts/wmdp_ta_sweep_phi-1_5.sh              # TaskArithmetic scale sweep (WMDP)
bash scripts/civil_comments_finetune_phi-1_5.sh    # prereqs: M1 + M_forget checkpoints (Civil Comments)
bash scripts/civil_comments_unlearn_phi-1_5.sh     # all methods on Civil Comments toxicity (subset training)
bash scripts/civil_comments_ta_sweep_phi-1_5.sh    # TaskArithmetic scale sweep (Civil Comments)
```

**Collect results** (aggregate eval outputs into Excel):
```bash
python scripts/collect_tofu_results.py saves/ --output results.xlsx
python scripts/collect_civil_comments_results.py saves/ --output civil_comments_results.xlsx
```

**Data setup**:
```bash
python setup_data.py --eval_logs      # Download pre-computed eval logs
python setup_data.py --idk            # Download IDK dataset variant
python setup_data.py --wmdp           # Download WMDP dataset
python setup_data.py --civil_comments # Download and filter Civil Comments dataset
```

## Architecture

All configuration is managed via **Hydra** YAML files in `configs/`. The two main entry points are `src/train.py` and `src/eval.py`, which instantiate components from the registry using config.

### Plugin Registry Pattern

Every major component type (trainer, dataset, model, collator, evaluator) uses a registry. To add a new component: implement a handler class, register it with `@registry.register(...)`, and create a Hydra config YAML. No core code changes required. See `docs/components.md` for detailed instructions.

### Component Hierarchy

**`src/trainer/`** — Training algorithms
- `base.py`: `FinetuneTrainer` extends HuggingFace `Trainer` with custom evaluator hooks
- `unlearn/`: 12+ unlearning method implementations (GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL, CEU, SatImp, WGA, PDU, TVD, AltPO) — all extend `UnlearnTrainer`
- `UnlearnTrainer` handles DeepSpeed, distributed training, and forget/retain batch splitting

**`src/evals/`** — Evaluation framework
- Benchmark evaluators: `TOFUEvaluator`, `MUSEEvaluator`, `LMEvalEvaluator`
- `metrics/`: memorization (verbatim prob, ROUGE, QA-ROUGE), privacy (6 MIA attacks), utility (forget quality, truth ratio)
- `mia/`: membership inference attack implementations (LOSS, ZLib, Reference, GradNorm, MinK, MinK++)
- Evaluators run during training (via `FinetuneTrainer`) and standalone via `eval.py`

**`src/data/`** — Dataset handlers
- `QADataset`: question-answering format (TOFU)
- `PretrainingDataset`: next-token prediction format (MUSE, WMDP); supports `max_samples` + `shuffle_seed` for reproducible corpus subsets
- `ForgetRetainDataset`: combined forget+retain splits required by most unlearning methods

**`src/model/`** — Model loading
- Wraps HuggingFace `AutoModelForCausalLM` with custom handlers
- `ProbedLlamaForCausalLM`: supports representation engineering (used by RMU/UNDIAL)

### Config Structure

```
configs/
├── train.yaml / unlearn.yaml / eval.yaml        # Top-level defaults
├── experiment/unlearn/{tofu,muse,wmdp}/          # Pre-configured full pipelines
├── experiment/finetune/{tofu,muse,wmdp}/         # Finetune prerequisite configs
├── experiment/eval/{tofu,muse,wmdp}/
├── trainer/                                      # One YAML per unlearning method
├── model/                                        # Supported model configs
├── data/datasets/                                # Dataset configs
└── eval/                                         # Evaluator/metric configs
```

Experiment configs (`configs/experiment/`) are the canonical way to run full pipelines — they compose trainer + data + model + eval configs.

### Supported Components

- **Benchmarks**: TOFU, MUSE (News/Books), WMDP (Bio/Cyber)
- **Methods**: GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL, CEU, SatImp, WGA, PDU, TVD, AltPO, TaskArithmetic
- **Models**: Llama-3.2 (1B/3B), Llama-3.1 (8B), Llama-2 (7B), Phi-3.5, Phi-1.5, Gemma, Zephyr

## Key Docs

- `docs/components.md` — How to add new trainers, datasets, evaluators, models
- `docs/experiments.md` — Running and reproducing experiments
- `docs/evaluation.md` — Evaluation framework details
- `community/leaderboard.md` — Results across all methods and benchmarks
