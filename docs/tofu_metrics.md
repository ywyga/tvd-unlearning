# TOFU Evaluation Metrics

This document describes each metric computed by the TOFU evaluator
(`configs/eval/tofu.yaml`) and what constitutes a good result after unlearning.

## Quick reference

| Metric | Better after unlearning | Requires retain logs |
|---|---|---|
| `forget_Truth_Ratio` | High | No |
| `forget_Q_A_Prob` | Low | No |
| `forget_Q_A_ROUGE` | Low | No |
| `extraction_strength` | Low | No |
| `forget_quality` | High | Yes |
| `privleak` | High | Yes |
| `model_utility` | High | No |

---

## Forgetting metrics

These measure how much the model has forgotten the content in the forget split.
Lower is better for all of them.

### `forget_Truth_Ratio`

**What it measures:** For each forget question, the model is given both the correct
answer and a paraphrased (wrong) answer. The truth ratio is the ratio of the
model's probability for the correct answer vs. the perturbed answer.

**Interpretation:** The aggregator is `mean(min(p_wrong/p_correct, p_correct/p_wrong))`,
which equals 1 when the model is indifferent between the correct and wrong answer, and
approaches 0 when it strongly prefers either one. A well-unlearned model should assign
similar probability to the correct and wrong answers on the forget set, giving a value
close to 1.

**Better = High**

---

### `forget_Q_A_Prob`

**What it measures:** The token-level probability the model assigns to the correct
answer when prompted with the forget question.

**Interpretation:** Direct measure of how likely the model is to reproduce the
forgotten content. A low value means the model has lost confidence in the correct
answer.

**Better = Low**

---

### `forget_Q_A_ROUGE`

**What it measures:** ROUGE-L recall score between the model's generated answer
and the ground-truth answer on forget questions. Uses free-form generation
(not forced teacher-forcing).

**Interpretation:** Measures whether the model actively reproduces the forgotten
content when prompted. Low ROUGE means the generated text does not match the
original answer.

**Better = Low**

---

### `extraction_strength`

**What it measures:** How much of the forget content can be extracted by prompting
with perturbed versions of the forget questions. Compares generation overlap
between perturbed and original question prompts.

**Interpretation:** Even when the exact question is not asked, a model that has
memorized content may still leak it through related prompts. Low extraction
strength means the forgotten knowledge is not recoverable through prompt variation.

**Better = Low**

---

## Privacy metrics

These require a reference evaluation log from a retain-finetuned model
(`retain_logs_path`). They are skipped (set to `null`) when the log is absent.

### `forget_quality`

**What it measures:** A Kolmogorov-Smirnov (KS) test comparing the distribution
of `forget_Truth_Ratio` values on the forget split between the unlearned model and
a model trained only on the retain split (i.e., a model that never saw the forget
data).

**Interpretation:** A high p-value means the two distributions are statistically
indistinguishable — the unlearned model behaves as if it was never trained on the
forget data. This is the gold-standard unlearning criterion in TOFU.

**Better = High** (p-value; typically reported as pass/fail at p > 0.05)

---

### `privleak`

**What it measures:** A Min-K% membership inference attack (MIA) on the forget
split, referenced against the retain model's MIA scores. Measures how
distinguishable forget samples are as training members compared to a model that
never trained on them.

**Interpretation:** A score near 0 means the unlearned model leaks no more
membership information about the forget data than the retain-only baseline. A
high positive score means forget samples are still recognizable as training members
— the unlearning is incomplete from a privacy standpoint.

**Better = High** (score closer to 0% leakage relative to the retain baseline)

---

## Utility metric

### `model_utility`

**What it measures:** Harmonic mean of 9 sub-metrics across three data splits:

| Split | Metrics |
|---|---|
| Retain split | `retain_Q_A_Prob`, `retain_Q_A_ROUGE`, `retain_Truth_Ratio` |
| Real-authors (ra) split | `ra_Q_A_Prob` (normalised), `ra_Q_A_ROUGE`, `ra_Truth_Ratio` |
| World-facts (wf) split | `wf_Q_A_Prob` (normalised), `wf_Q_A_ROUGE`, `wf_Truth_Ratio` |

The harmonic mean penalises any single sub-metric that collapses, making it a
strict test of whether the model retains general knowledge after unlearning.

**Interpretation:** Captures the cost of unlearning on the model's overall
capability. A drop in utility indicates that the unlearning method damaged
knowledge beyond the intended forget set.

**Better = High**

---

## The core trade-off

Forgetting metrics and utility pull in opposite directions. Aggressive unlearning
methods push the forgetting metrics down but tend to reduce `model_utility`.
A good unlearning method achieves low forgetting metrics and high utility
simultaneously. `forget_quality` (when available) provides the cleanest single
criterion: it asks whether the model is statistically indistinguishable from one
that never saw the forget data.
