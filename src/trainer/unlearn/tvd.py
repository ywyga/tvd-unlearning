import copy
import os
import torch
from transformers import AutoModelForCausalLM
from trainer.unlearn.base import UnlearnTrainer


class TVD(UnlearnTrainer):
    """
    Task Vector Decomposition (TVD) unlearning method.

    Starting from M1 (fine-tuned on the full dataset), jointly trains:
      - M_retain: optimised on the retain split  ->  TV_retain = M_retain - M0
      - M_forget: optimised on the forget split  ->  TV_forget = M_forget - M0

    where M0 is the frozen base model and TV1 = M1 - M0 is the full task vector.

    Loss:
        L_total = lambda_data        * L_data
                + lambda_reconstruct * L_reconstruct   (TV_retain + TV_forget ≈ TV1)
                + lambda_orth        * L_orth           (TV_retain ⊥ TV_forget)
                + lambda_norm        * L_norm           (||TV_retain|| ≈ ||TV1||)

    Final unlearned model: M0 + TV_retain  (= self.model after training).
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        lambda_reconstruct: float = 1.0,
        lambda_data: float = 1.0,
        lambda_orth: float = 1.0,
        lambda_norm: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_data = lambda_data
        self.lambda_orth = lambda_orth
        self.lambda_norm = lambda_norm

        # --- Load M0 (frozen base model) ---
        # M0 is loaded from the provided path and kept frozen throughout training.
        m0 = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=self.model.dtype,
        ).to(self.accelerator.device)
        m0.eval()
        for p in m0.parameters():
            p.requires_grad_(False)

        # --- Compute TV1 = M1 - M0 and store M0 params, both on CPU ---
        # Must be computed BEFORE wrapping m0 with DeepSpeed/accelerator, because
        # ZeRO-3 shards parameters across ranks — accessing .parameters() after
        # wrapping returns partial tensors that don't match self.model's full shapes.
        # M0 params are kept on CPU (moved to device per-tensor in compute_loss)
        # to avoid occupying a full model's worth of GPU memory with a frozen copy.
        with torch.no_grad():
            self.tv1 = []
            self.m0_params = []
            for p_m1, p_m0 in zip(self.model.parameters(), m0.parameters()):
                self.tv1.append((p_m1.data - p_m0.data).cpu())
                self.m0_params.append(p_m0.data.cpu())
        # m0 is no longer needed on GPU; let it go out of scope to free memory.

        # Precompute ||TV1|| as a plain float for the scale-invariant L_norm.
        self.tv1_norm: float = (
            torch.sqrt(sum(v.pow(2).sum() for v in self.tv1)).item()
        )

        # --- Create M_forget: second trainable model initialised from M1 ---
        self.forget_model = copy.deepcopy(self.model).to(self.accelerator.device)
        self.forget_model.train()

    def create_optimizer(self):
        """Extend the default HF optimizer to jointly optimise M_forget."""
        super().create_optimizer()
        forget_params = [p for p in self.forget_model.parameters() if p.requires_grad]
        self.optimizer.add_param_group(
            {"params": forget_params, "lr": self.args.learning_rate}
        )
        return self.optimizer

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        save_dir = output_dir or self.args.output_dir
        self.forget_model.save_pretrained(os.path.join(save_dir, "forget_model"))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        retain_inputs = {
            k: inputs["retain"][k]
            for k in ["input_ids", "attention_mask", "labels"]
        }
        forget_inputs = {
            k: inputs["forget"][k]
            for k in ["input_ids", "attention_mask", "labels"]
        }

        # Unwrap the Trainer/Accelerate model wrapper so the forward call
        # behaves identically to forget_model (a plain deepcopy). Without this,
        # the wrapper returns a partial loss tensor (e.g. 7 elements from a
        # 14-sample batch) instead of the standard mean-reduced scalar.
        base_model = getattr(model, "module", model)

        # Forward passes for both trainable models
        retain_outputs = base_model(**retain_inputs)         # M_retain on retain data
        forget_outputs = self.forget_model(**forget_inputs)  # M_forget on forget data

        # L_data: combined NLL on retain / forget data
        l_data = retain_outputs.loss.mean() + forget_outputs.loss.mean()

        device = next(model.parameters()).device

        # Compute all task-vector losses in a single parameter loop.
        # Avoids building tv_retain/tv_forget lists and, critically, avoids
        # torch.cat over all params (which would create two full-model-sized
        # contiguous tensors just to compute a dot product and two norms).
        # M0 params live on CPU and are moved to device one tensor at a time.
        l_reconstruct = torch.tensor(0.0, device=device)
        dot = torch.tensor(0.0, device=device)
        retain_norm_sq = torch.tensor(0.0, device=device)
        forget_norm_sq = torch.tensor(0.0, device=device)
        n_layers = 0

        for param_r, param_f, m0_p, tv1_p in zip(
            model.parameters(),
            self.forget_model.parameters(),
            self.m0_params,
            self.tv1,
        ):
            ref = m0_p.to(device)
            tr = param_r - ref
            tf = param_f - ref
            l_reconstruct = l_reconstruct + (tr + tf - tv1_p.to(device)).pow(2).mean()
            dot = dot + (tr * tf).sum()
            retain_norm_sq = retain_norm_sq + tr.pow(2).sum()
            forget_norm_sq = forget_norm_sq + tf.pow(2).sum()
            n_layers += 1

        l_reconstruct = l_reconstruct / n_layers

        retain_norm = retain_norm_sq.sqrt()
        forget_norm = forget_norm_sq.sqrt()

        # L_orth: squared cosine similarity — drives TV_retain ⊥ TV_forget
        cos_sim = dot / (retain_norm * forget_norm + 1e-8)
        l_orth = cos_sim.pow(2)

        # L_norm: scale-invariant penalty — (||TV_retain|| / ||TV1|| - 1)^2
        l_norm = (retain_norm / (self.tv1_norm + 1e-8) - 1.0).pow(2)

        loss = (
            self.lambda_data * l_data
            + self.lambda_reconstruct * l_reconstruct
            + self.lambda_orth * l_orth
            + self.lambda_norm * l_norm
        )

        # Log individual loss components to TensorBoard (PDU pattern, pdu.py:65-71)
        self.log({
            "loss_reconstruct": l_reconstruct.item(),
            "loss_data": l_data.item(),
            "loss_orth": l_orth.item(),
            "loss_norm": l_norm.item(),
        })

        return (loss, retain_outputs) if return_outputs else loss
