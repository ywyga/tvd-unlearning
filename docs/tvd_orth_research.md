# TVD Orthogonality Loss: Research Notes and Improvement Ideas

This document surveys the literature relevant to improving the orthogonality penalty (`L_orth`) in Task Vector Decomposition (TVD) unlearning, examines how related methods in the codebase handle analogous constraints, and proposes concrete alternatives to the current global cosine-similarity penalty.

---

## 1. Relevant Papers

### 1.1 Task Arithmetic — Editing Models with Task Vectors (Ilharco et al., ICLR 2023)

**Paper:** [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)

**Mechanism.** Task vectors are defined as `τ = θ_finetuned − θ_pretrained`. The paper shows that multiple task vectors can be added together with minimal interference — and proves theoretically that interference is minimized when task vectors are pairwise orthogonal. In practice, independently fine-tuned task vectors are already approximately orthogonal, which is why simple arithmetic (addition, negation) works well.

**Orthogonality scope.** Global parameter-space dot product, identical to what TVD currently uses.

**Relevance to TVD.** TVD's L_orth directly inherits the cosine-penalty framing from this line of work. The key empirical observation from the paper — that independently learned task vectors tend toward orthogonality naturally — raises the question of whether the soft penalty is sufficient or whether harder constraints would produce cleaner decompositions.

---

### 1.2 Gradient Surgery for Multi-Task Learning — PCGrad (Yu et al., NeurIPS 2020)

**Paper:** [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)

**Mechanism.** When two tasks' gradients have negative cosine similarity (i.e., they conflict), PCGrad projects each task's gradient onto the plane orthogonal to the other task's gradient before the update step:

```
if cos(g_i, g_j) < 0:
    g_i ← g_i − (g_i · g_j / ||g_j||²) * g_j
```

This is a **hard projection** applied per gradient update, not a soft penalty added to the loss. No hyperparameter λ controls the strength — the projection either fires or does not.

**Orthogonality scope.** Gradient space, applied globally across all parameters (a single flat gradient vector per task).

**Relevance to TVD.** TVD's current L_orth is a soft penalty on the task-vector dot product in weight space, which is static relative to a given parameter snapshot. PCGrad instead enforces orthogonality dynamically at every gradient step. One could apply PCGrad-style surgery between M_retain's and M_forget's gradients during TVD training — removing any component of the retain gradient that points toward the forget gradient, and vice versa — as a supplement or replacement for the penalty term.

---

### 1.3 TIES-Merging (Yadav et al., NeurIPS 2023)

**Paper:** [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)

**Mechanism.** Identifies two sources of interference when merging task vectors: (a) redundant near-zero parameters and (b) sign conflicts across models. The procedure: (1) **trim** — zero out task-vector entries below a magnitude threshold; (2) **elect sign** — resolve conflicts by majority vote; (3) **disjoint merge** — average only parameters that agree with the elected sign.

This is effectively a hard, sparse form of interference reduction: parameters that conflict are simply excluded from the merge.

**Orthogonality scope.** Per-parameter sign-based filtering, not a continuous orthogonality constraint.

**Relevance to TVD.** TIES suggests that the small-magnitude entries in a task vector are primarily noise that causes interference without contributing to task performance. A TVD variant could apply magnitude-based trimming to TV_retain and TV_forget before computing the orthogonality penalty — or could use sign-disagreement as an additional regularizer signal (parameters where TV_retain and TV_forget have the same sign are likely to interfere, and could be penalized more heavily).

---

### 1.4 DARE (Yu et al., 2023)

**Paper:** [Language Models are Super Mario](https://arxiv.org/abs/2311.03099) (introduces DARE)

**Mechanism.** Randomly drops (zeros out) a high fraction (up to 90%) of delta parameters (task vectors) and rescales the survivors by `1 / (1 − p)` to preserve expected magnitude. This stochastic sparsification dramatically reduces interference during merging because sparse vectors are much more likely to be approximately orthogonal.

**Orthogonality scope.** Stochastic, parameter-level sparsity — not an explicit orthogonality constraint but achieves near-orthogonality as a byproduct of sparsity.

**Relevance to TVD.** Applying DARE-style dropout to TV_retain and TV_forget during training would automatically reduce the cosine similarity between them without a dedicated penalty term. However, the random masking would need to be consistent within a training step to avoid conflicting gradient signals.

---

### 1.5 AdaMerging (Yang et al., ICLR 2024)

**Paper:** [AdaMerging: Adaptive Model Merging for Multi-Task Learning](https://arxiv.org/abs/2310.02575)

**Mechanism.** Learns per-layer (or per-task) scalar coefficients for task-vector contributions via entropy minimization on unlabeled test data. The layer-wise variant assigns a separate coefficient `λ_l` to each layer's task vector, allowing high-interference layers to be down-weighted automatically.

**Orthogonality scope.** Implicitly per-layer: the learned coefficients adapt to layer-level interference, but without an explicit orthogonality constraint.

**Relevance to TVD.** Motivates the idea of per-layer orthogonality loss weighting in TVD. Layers that carry more of the task signal (large ||TV1_l||) or exhibit higher forget/retain interference may deserve a stronger orthogonality penalty. AdaMerging's success with layer-wise coefficients supports treating the network's layers heterogeneously rather than with a single global penalty.

---

### 1.6 Orthogonal Gradient Descent — OGD (Farajtabar et al., AISTATS 2020)

**Paper:** [Orthogonal Gradient Descent for Continual Learning](https://arxiv.org/abs/1910.07104)

**Mechanism.** Maintains a basis for the gradient subspace of previous tasks (accumulated via SVD of past gradient matrices). For each new task update, projects the gradient onto the complement of this subspace, guaranteeing that the network output on previous tasks is unchanged to first order.

**Orthogonality scope.** Hard projection in gradient space; operates globally but with an accumulated subspace rather than a single reference vector.

**Relevance to TVD.** OGD's subspace projection is the "hard" version of what TVD's L_orth does softly. In the TVD setting, the "previous task" is the retain set. At each training step, one could project M_retain's gradient onto the subspace orthogonal to M_forget's recent gradient (or the accumulated gradient basis of the forget set), ensuring that the retain model's updates do not encode forget-set information. This is conceptually tighter than a global cosine penalty because it operates incrementally at every gradient step rather than penalizing end-state similarity.

---

### 1.7 Projected-Gradient Unlearning — PGU (Hoang et al., WACV 2024)

**Paper:** [Learn to Unlearn for Deep Neural Networks: Minimizing Unlearning Interference with Gradient Projection](https://arxiv.org/abs/2312.04095)

**Mechanism.** Partitions gradient space into a Core Gradient Space (CGS) — principal gradient subspace of the retain set — and its complement (Residual Gradient Space). The forget update is projected onto the RGS so that it cannot damage retain performance. Implemented via SVD of the retain gradient matrix.

**Orthogonality scope.** Hard projection in gradient space; the retain subspace is computed from the retain data's gradients, making it data-driven.

**Relevance to TVD.** PGU directly addresses TVD's core problem (forget updates interfering with retain performance) and does so without a soft penalty. Translating PGU into TVD: at each step, compute the top-k left singular vectors of the stacked per-sample retain gradients, project the forget model's gradient update onto their orthogonal complement, and update M_forget with the projected gradient. This eliminates L_orth as a penalty term entirely and replaces it with a hard structural constraint.

---

### 1.8 OrthoGrad — Per-Sample Gradient Orthogonalization (2025)

**Paper:** [Go Beyond Your Means: Unlearning with Per-Sample Gradient Orthogonalization](https://arxiv.org/abs/2503.02312)

**Mechanism.** Projects the aggregated forget/unlearn gradient onto the subspace orthogonal to **each individual retain sample's gradient** (not just the batch-mean retain gradient). This per-sample treatment is shown to reduce both mean error and its variance compared to batch-level projection.

**Orthogonality scope.** Hard projection, per-sample, in gradient space.

**Relevance to TVD.** OrthoGrad is the strongest available evidence that hard per-sample gradient projection outperforms soft penalties for unlearning. For TVD, a practical approximation would be: accumulate per-sample retain gradients (using hooks or HuggingFace's per-sample gradient support), estimate the retain subspace, and project the forget gradient onto its complement before the optimizer step.

---

### 1.9 Ethos: Rectifying LMs in Orthogonal Parameter Space (Gao et al., NAACL 2024)

**Paper:** [Ethos: Rectifying Language Models in Orthogonal Parameter Space](https://arxiv.org/abs/2403.08994)

**Mechanism.** Applies SVD to weight matrices of the pre-trained model to extract principal components. Classifies components as "general" (to preserve) or "undesired" (to remove) based on their activation patterns on target data. Performs unlearning by negating only the undesired components' task vector, leaving general components intact.

**Orthogonality scope.** Per-layer SVD decomposition; operates on the weight matrix of each transformer layer independently.

**Relevance to TVD.** Ethos demonstrates that decomposing task vectors layer-by-layer in their singular-vector basis isolates forget-relevant components from retain-relevant ones more cleanly than global parameter-space operations. For TVD, applying SVD to each layer's TV1_l and computing orthogonality penalties in the principal-component subspace rather than in the raw parameter space could produce a more semantically meaningful decomposition.

---

### 1.10 Per-Parameter Task Arithmetic — PerTA (2025)

**Paper:** [Per-parameter Task Arithmetic for Unlearning in Large Language Models](https://arxiv.org/abs/2601.22030)

**Mechanism.** Replaces the scalar task-vector scaling coefficient with per-parameter importance weights, estimated via raw gradients (PerTA-grad) or diagonal Fisher information (PerTA-fisher). Parameters with high importance for retention are down-weighted in the forget task vector; parameters more diagnostic of the forget set are up-weighted.

**Orthogonality scope.** Per-parameter importance weighting; no explicit orthogonality constraint, but effectively decouples retain and forget parameter contributions.

**Relevance to TVD.** PerTA's Fisher-based importance weights are directly applicable to TVD's L_orth: instead of a flat global dot product, one could compute a **Fisher-weighted cosine similarity** between TV_retain and TV_forget, assigning higher penalty to parameter dimensions that are jointly important to both sets. This focuses the orthogonality constraint on the parameters that matter most.

---

### 1.11 Task Singular Vectors (TSV) for Interference Reduction (2024)

**Paper:** [Task Singular Vectors: Reducing Task Interference in Model Merging](https://arxiv.org/abs/2412.00081)

**Mechanism.** Decomposes each layer's task-vector matrix via SVD. Uses the singular vectors (not just scalar magnitudes) as the unit of orthogonalization. Decorrelates singular vectors across tasks to minimize per-layer interference.

**Orthogonality scope.** Per-layer SVD, operating on the geometric structure of weight matrices as rank-r objects rather than flat vectors.

**Relevance to TVD.** Most directly applicable to the per-layer variant of L_orth: instead of computing `dot(flatten(TV_retain_l), flatten(TV_forget_l))`, decompose each into its dominant left singular vectors and penalize the overlap between those subspaces (using, e.g., the Frobenius norm of the product of left singular matrices). This is more sensitive to the directional structure of weight-matrix updates than a flat dot product.

---

## 2. How RMU and UNDIAL Handle Related Constraints

### 2.1 RMU (Representation Misdirection for Unlearning)

**File:** `src/trainer/unlearn/rmu.py`

**What it does.** RMU operates entirely in **activation (representation) space**, not parameter space. It:

1. Selects a single transformer layer (default: layer 7) via `module_regex` and registers a forward hook to capture its output activations during the forward pass.
2. For the **forget set**: computes MSE between the model's activations at that layer and a fixed random unit vector scaled by `steering_coeff` (the "control vector"). This pushes the model to produce arbitrary, meaningless activations for forget inputs — effectively misdirecting its internal representations.
3. For the **retain set**: computes MSE between the model's activations at the chosen layer and the *reference model's* activations at the same layer (using `retain_loss_type == "EMBED_DIFF"`). This pins the model's representations on retain inputs to match the original model.
4. Only a small subset of parameters (3 MLP down-projection weights in layers 5–7) are trainable.

**Comparison to TVD.** RMU enforces orthogonality implicitly at the representation level: forget activations are pushed toward a random direction while retain activations are anchored to the reference model. There is no explicit parameter-space dot product. The constraint is **layer-specific** (a single layer) and **representation-level** (activations, not weights). The random control vector ensures the forget direction is arbitrary and thus unlikely to align with any retain direction — a practical substitute for orthogonality without formal enforcement.

**What TVD could borrow.** The idea of anchoring intermediate representations of M_retain to the reference model M1 at specific layers — as a complement to the parameter-space L_orth — adds a representation-space regularizer that prevents retain drift beyond what L_norm alone captures.

---

### 2.2 UNDIAL (Unlearning via Diagonal Intervention on Adjusted Logits)

**File:** `src/trainer/unlearn/undial.py` + `src/trainer/utils.py:compute_undial_loss`

**What it does.** UNDIAL operates in **output logit space**. For the forget set:

1. Runs the reference model (frozen M1) on the forget input to get teacher logits.
2. Constructs an "adjusted teacher" distribution by subtracting a scalar `beta` from the logit of the ground-truth token, then applies softmax. This creates a soft label that down-weights the correct answer without fully inverting it.
3. Trains the student model (M_retain equivalent) to match these adjusted soft labels via cross-entropy.

For the retain set, UNDIAL falls back to the standard GradDiff retain loss (KL or NLL against the reference).

**Comparison to TVD.** UNDIAL's constraint is entirely in **output distribution space** — it never touches parameter geometry. There is no orthogonality term at all. The interference control is implicit: by targeting only the logit of the correct token (via the diagonal mask `mask[..., shift_labels.unsqueeze(-1)] = 1.0`), UNDIAL limits forget-set influence to the minimal output-level change. This is complementary to TVD's parameter-space decomposition: UNDIAL controls *what* the model outputs; TVD controls *how* the weight updates are structured.

**What TVD could borrow.** The concept of operating on the KL-divergence between adjusted distributions rather than a hard NLL objective for the forget model — UNDIAL's soft-label approach might provide a more stable forget objective inside TVD than the raw NLL loss currently used for M_forget.

---

### 2.3 Summary Comparison

| Method | Constraint domain | Level | Hard or soft | Reference model |
|--------|-------------------|-------|-------------|-----------------|
| TVD (current) | Weight/parameter space | Global flat | Soft (penalty) | M0 (base) |
| RMU | Activation space | Single layer | Soft (MSE) | M1 (fine-tuned) |
| UNDIAL | Logit/output space | Per-token | Soft (KL-like) | M1 (fine-tuned) |
| PCGrad | Gradient space | Global flat | Hard (projection) | None |
| OGD / PGU | Gradient space | Subspace (SVD) | Hard (projection) | Retain data |
| OrthoGrad | Gradient space | Per-sample | Hard (projection) | Retain samples |
| Ethos | Per-layer SVD | Layer-wise | Hard (selection) | Pre-trained SVD |

---

## 3. Concrete Improvement Ideas for L_orth

### Idea A: Per-Layer Orthogonality Loss (Drop-in Replacement)

**Motivation.** The current L_orth flattens all parameters into two global vectors and computes a single cosine similarity. This treats every layer equally, but transformer layers differ dramatically in their parameter count, gradient magnitude, and task relevance. A 4096×4096 attention weight contributes far more to the dot product than a 4096-element bias purely due to dimensionality.

**Implementation.** Replace the single accumulated `dot` with a sum of per-layer cosine similarities, optionally weighted by each layer's parameter count or TV1 magnitude:

```python
l_orth = torch.tensor(0.0, device=device)
total_weight = 0.0

for param_r, param_f, m0_p, tv1_p in zip(...):
    tr = param_r - ref
    tf = param_f - ref
    dot_l = (tr * tf).sum()
    norm_r_l = tr.pow(2).sum().sqrt()
    norm_f_l = tf.pow(2).sum().sqrt()
    cos_l = dot_l / (norm_r_l * norm_f_l + 1e-8)
    # weight by TV1 contribution of this layer
    weight = tv1_p.pow(2).sum().item()
    l_orth = l_orth + weight * cos_l.pow(2)
    total_weight += weight

l_orth = l_orth / (total_weight + 1e-8)
```

**Properties.** Same computational cost as the current implementation (single parameter loop). Naturally focuses the orthogonality pressure on layers that carry the most of the original task signal. No new hyperparameters required.

**Interaction with L_reconstruct.** L_reconstruct already implicitly weights layers by their MSE contribution, so this change makes L_orth consistent with L_reconstruct's implicit weighting scheme.

---

### Idea B: PCGrad-Style Gradient Surgery Between M_retain and M_forget

**Motivation.** The current L_orth penalizes the *end-of-step* similarity of the task vectors, but does not control the *direction of gradient updates* during training. PCGrad shows that projecting conflicting gradients is more effective than adding a penalty because it acts before the parameter update.

**Implementation.** After the backward pass (before `optimizer.step()`), for each parameter pair `(p_r, p_f)`:

```python
for p_r, p_f in zip(model.parameters(), forget_model.parameters()):
    if p_r.grad is None or p_f.grad is None:
        continue
    g_r = p_r.grad.data.view(-1)
    g_f = p_f.grad.data.view(-1)
    cos = (g_r @ g_f) / (g_r.norm() * g_f.norm() + 1e-8)
    if cos < 0:
        # project g_r to remove component along g_f
        p_r.grad.data -= (g_r @ g_f / (g_f.norm()**2 + 1e-8)) * p_f.grad.data
        # and symmetrically
        p_f.grad.data -= (g_f @ g_r / (g_r.norm()**2 + 1e-8)) * p_r.grad.data
```

This hook would be registered in `TVD.__init__` or applied in a custom `training_step` override. The `lambda_orth` weight could be zeroed out when gradient surgery is active.

**Properties.** Hard constraint — orthogonality in gradient space is enforced at every step, not just penalized in the loss. Only fires when gradients conflict (negative cosine), so it does not disturb aligned gradients. Eliminates the need to tune `lambda_orth`.

**Caveat.** Per-parameter surgery over a full LLM is computationally expensive at each step. A practical approximation: compute the global dot product of the flattened retain/forget gradient vectors (already available implicitly from the L_orth backward pass) and apply a single projection to the flat vector rather than layer-by-layer. This preserves the O(P) cost of the current implementation.

---

### Idea C: Subspace Projection via Retain Gradient SVD (PGU-Inspired Hard Constraint)

**Motivation.** PGU and OGD show that projecting forget-set gradient updates onto the complement of the retain gradient subspace is more principled than a cosine penalty, because it guarantees (to first order) that the forget update does not degrade retain performance.

**Implementation.** At the start of each training step (or every K steps), compute the top-k left singular vectors of the stacked per-sample retain gradients to form a retain subspace basis U_r (shape: P × k, where P = total parameters). Then project M_forget's gradient:

```python
g_f_flat = torch.cat([p.grad.view(-1) for p in forget_model.parameters()])
# project onto complement of retain subspace
proj = U_r @ (U_r.T @ g_f_flat)   # component in retain subspace
g_f_flat = g_f_flat - proj          # remove retain subspace component
# scatter back
offset = 0
for p in forget_model.parameters():
    numel = p.numel()
    p.grad.data.copy_(g_f_flat[offset:offset+numel].view_as(p.grad))
    offset += numel
```

**Properties.** Hard constraint in the strongest available sense — the forget update is guaranteed to be orthogonal to the retain gradient subspace. Subsumes L_orth: if adopted, `lambda_orth` should be set to 0. The main cost is computing per-sample gradients and their SVD, which is O(B * P * k) per step.

**Practical approximation.** For LLM scale, computing exact per-sample gradients is expensive. A cheaper variant: use the gradient of the batch-mean retain loss (already computed in the current TVD forward pass) as a rank-1 approximation of the retain subspace, then project M_forget's gradient to be orthogonal to this single vector. This is equivalent to PCGrad but framed as projection rather than conditional surgery.

---

### Idea D: Fisher-Weighted Orthogonality Penalty (PerTA-Inspired)

**Motivation.** PerTA shows that parameters differ widely in their importance for retain vs. forget sets. A flat dot product weights all parameters equally; a Fisher-weighted dot product concentrates the orthogonality penalty on the parameters that matter most.

**Implementation.** Precompute diagonal Fisher information on retain and forget sets:

```python
F_retain[i] = E[(∂ log p(y|x) / ∂ θ_i)²]   # approximated by squared gradients
F_forget[i]  = similarly for forget set
W[i] = F_retain[i] * F_forget[i]             # high when parameter matters to both
```

Replace the L_orth dot product with:

```python
# weighted dot product
w = (F_retain * F_forget).sqrt()  # element-wise geometric mean
dot_weighted = ((tr * tf) * w).sum()
norm_r_w = ((tr.pow(2)) * w).sum().sqrt()
norm_f_w = ((tf.pow(2)) * w).sum().sqrt()
l_orth = (dot_weighted / (norm_r_w * norm_f_w + 1e-8)).pow(2)
```

**Properties.** No computational overhead at training time (Fisher estimates are precomputed once). Focuses orthogonality pressure on parameters that are jointly informative for both sets, which are exactly the ones where interference is most damaging. Fisher estimates could be computed once in `TVD.__init__` using a small sample of each split.

---

### Idea E: Per-Layer SVD Subspace Orthogonality (Ethos / TSV Inspired)

**Motivation.** Weight matrices in transformers are not arbitrary flat vectors — they are low-rank-structured matrices whose singular vectors encode semantic directions. Computing dot products on flattened matrices discards this structure. Ethos and TSV demonstrate that operating on singular vectors is more semantically meaningful.

**Implementation.** For each weight matrix layer (e.g., each attention projection or MLP weight), compute truncated SVD of TV_retain_l and TV_forget_l, then penalize the overlap between their top-k left singular vector subspaces:

```python
for (name_r, p_r), (name_f, p_f), m0_p in zip(...):
    if p_r.ndim < 2:    # skip bias terms, norms
        continue
    tr_l = (p_r - ref).view(p_r.shape[0], -1)   # shape: (d_out, d_in*...)
    tf_l = (p_f - ref).view(p_f.shape[0], -1)
    U_r, _, _ = torch.linalg.svd(tr_l, full_matrices=False)
    U_f, _, _ = torch.linalg.svd(tf_l, full_matrices=False)
    # Overlap between top-k singular subspaces: ||U_r[:,:k].T @ U_f[:,:k]||_F^2
    k = min(4, U_r.shape[1])
    overlap = torch.norm(U_r[:, :k].T @ U_f[:, :k], p='fro').pow(2)
    l_orth = l_orth + overlap
```

**Properties.** Captures interference at the level of the dominant geometric directions of weight-matrix updates, which is more semantically meaningful than a flat dot product. Per-layer by construction.

**Caveat.** SVD of every weight matrix at every training step is expensive. This is best applied every K steps (e.g., K=10) or approximated with a power-iteration method. For a 1B-parameter model with ~60 weight matrices, computing rank-4 SVD per step adds roughly 60 small SVDs of shape (4096, 4096) — feasible but not negligible.

---

### Idea F: Remove L_orth and Rely on L_reconstruct Implicitly

**Motivation.** The reconstruction constraint `TV_retain + TV_forget ≈ TV1` implies `TV_retain ≈ TV1 − TV_forget`. If L_norm simultaneously enforces `||TV_retain|| ≈ ||TV1||` and L_reconstruct is tight, then `TV_forget` is forced toward `TV1 − TV_retain`, which is already determined. It is worth asking whether L_orth provides independent information or is largely redundant.

**Analysis.** Given the identity `TV_retain + TV_forget = TV1` exactly, the cosine similarity is:

```
cos(TV_retain, TV_forget) = cos(TV_retain, TV1 - TV_retain)
    = (TV_retain · TV1 - ||TV_retain||²) / (||TV_retain|| * ||TV1 - TV_retain||)
```

When `||TV_retain|| ≈ ||TV1||`, this simplifies. Orthogonality of TV_retain and TV_forget requires:

```
TV_retain · TV_forget = 0
TV_retain · (TV1 - TV_retain) = 0
TV_retain · TV1 = ||TV_retain||²
```

This is a non-trivial additional constraint beyond reconstruction — it requires TV_retain to project TV1 exactly onto itself (i.e., TV_retain is the projection of TV1 onto the retain subspace). So L_orth is **not redundant** given L_reconstruct, but they are geometrically coupled. A perfectly tight L_reconstruct + L_orth would force TV_retain and TV_forget to be exactly the retain and forget components of a vector decomposition — which is the correct formulation.

**Practical implication.** L_orth should remain in the loss but its effective strength is modulated by how tightly L_reconstruct is satisfied. Monitoring the ratio `l_orth / l_reconstruct` during training can identify whether the orthogonality constraint is active or vacuous.

---

## 4. Recommendation: What to Try First

### Primary recommendation: Idea A (per-layer TV-magnitude weighted L_orth) + Idea D (Fisher weighting)

**Rationale:**

1. **Lowest implementation risk.** Both ideas are drop-in modifications to the existing parameter loop in `compute_loss`. No changes to the optimizer, trainer structure, or data pipeline. Can be validated with a single TOFU training run.

2. **Well-motivated by literature.** AdaMerging's success with per-layer coefficients (ICLR 2024) and PerTA's Fisher-weighting results (arXiv 2025) both demonstrate that heterogeneous treatment of layers/parameters strictly outperforms global flat operations for task-vector interference.

3. **Addresses the main weakness directly.** The current L_orth penalizes all parameters equally, so a large MLP weight matrix (millions of parameters) dominates the dot product and a small embedding layer (which may be semantically more critical) is effectively ignored. Per-layer weighting by `||TV1_l||²` makes L_orth sensitive to exactly the layers that carry the most unlearning-relevant signal.

4. **Complementary to the existing loss terms.** Idea A is consistent with L_reconstruct's implicit layer weighting (MSE per layer averaged across layers). Idea D adds a data-driven importance signal without requiring any new training infrastructure.

**Implementation sketch (combined A + partial D):**

```python
# In TVD.compute_loss, replace the current dot/retain_norm_sq/forget_norm_sq accumulators:

l_orth_num = torch.tensor(0.0, device=device)
l_orth_den_r = torch.tensor(0.0, device=device)
l_orth_den_f = torch.tensor(0.0, device=device)

for param_r, param_f, m0_p, tv1_p in zip(...):
    ref = m0_p.to(device)
    tr = param_r - ref
    tf = param_f - ref
    layer_weight = tv1_p.pow(2).sum()     # TV1 magnitude of this layer
    l_orth_num = l_orth_num + layer_weight * (tr * tf).sum()
    l_orth_den_r = l_orth_den_r + layer_weight * tr.pow(2).sum()
    l_orth_den_f = l_orth_den_f + layer_weight * tf.pow(2).sum()

cos_sim_weighted = l_orth_num / (l_orth_den_r.sqrt() * l_orth_den_f.sqrt() + 1e-8)
l_orth = cos_sim_weighted.pow(2)
```

This is a pure refactoring of the existing loop — the new weights `layer_weight` focus the penalty on the TV1-dominant layers, and the formula reduces to the original when all `layer_weight` are equal.

### Secondary recommendation: Idea B (gradient surgery / PCGrad between M_retain and M_forget)

If per-layer weighting alone proves insufficient, the next step is to add a gradient-space hard projection. Start with the cheapest form: a single rank-1 projection of the global retain gradient onto the complement of the global forget gradient (one extra forward-pass worth of memory, no SVD required). This is directly interpretable and has strong empirical backing from the PCGrad and OrthoGrad literature.

---

## 5. Sources

- [Editing Models with Task Arithmetic (Ilharco et al., ICLR 2023)](https://arxiv.org/abs/2212.04089)
- [Gradient Surgery for Multi-Task Learning / PCGrad (Yu et al., NeurIPS 2020)](https://arxiv.org/abs/2001.06782)
- [TIES-Merging: Resolving Interference When Merging Models (Yadav et al., NeurIPS 2023)](https://arxiv.org/abs/2306.01708)
- [AdaMerging: Adaptive Model Merging for Multi-Task Learning (Yang et al., ICLR 2024)](https://arxiv.org/abs/2310.02575)
- [Orthogonal Gradient Descent for Continual Learning / OGD (Farajtabar et al., AISTATS 2020)](https://arxiv.org/abs/1910.07104)
- [Learn to Unlearn for Deep Neural Networks: Minimizing Unlearning Interference with Gradient Projection / PGU (Hoang et al., WACV 2024)](https://arxiv.org/abs/2312.04095)
- [Go Beyond Your Means: Unlearning with Per-Sample Gradient Orthogonalization / OrthoGrad (2025)](https://arxiv.org/abs/2503.02312)
- [Ethos: Rectifying Language Models in Orthogonal Parameter Space (Gao et al., NAACL 2024)](https://arxiv.org/abs/2403.08994)
- [Per-parameter Task Arithmetic for Unlearning in Large Language Models / PerTA (2025)](https://arxiv.org/abs/2601.22030)
- [Task Singular Vectors: Reducing Task Interference in Model Merging / TSV (2024)](https://arxiv.org/abs/2412.00081)
