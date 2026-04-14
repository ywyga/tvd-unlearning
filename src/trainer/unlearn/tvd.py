import copy
import torch
import torch.nn.functional as F
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

        if self.is_deepspeed_enabled:
            self.ref_model = self._prepare_deepspeed(m0)
        else:
            self.ref_model = self.accelerator.prepare_model(m0, evaluation_mode=True)

        # --- Compute TV1 = M1 - M0 (fixed reference vector) ---
        # Stored as a list of CPU tensors to avoid occupying additional GPU memory.
        with torch.no_grad():
            self.tv1 = [
                (p_m1.data - p_m0.data).cpu()
                for p_m1, p_m0 in zip(
                    self.model.parameters(), self.ref_model.parameters()
                )
            ]

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

        # Forward passes for both trainable models
        retain_outputs = model(**retain_inputs)           # M_retain on retain data
        forget_outputs = self.forget_model(**forget_inputs)  # M_forget on forget data

        # L_data: combined NLL on retain / forget data
        l_data = retain_outputs.loss + forget_outputs.loss

        # Build differentiable task vectors TV_retain and TV_forget.
        # Parameters are iterated positionally across all three models.
        # ref_param is detached so gradients only flow through param_r / param_f.
        tv_retain, tv_forget = [], []
        for param_r, param_f, ref_param in zip(
            model.parameters(),
            self.forget_model.parameters(),
            self.ref_model.parameters(),
        ):
            ref = ref_param.detach()
            tv_retain.append(param_r - ref)
            tv_forget.append(param_f - ref)

        device = tv_retain[0].device

        # L_reconstruct: ||TV_retain + TV_forget - TV1||^2
        # Averaged element-wise over all parameter tensors to be scale-neutral.
        l_reconstruct = torch.stack([
            (tr + tf - tv1_p.to(device)).pow(2).mean()
            for tr, tf, tv1_p in zip(tv_retain, tv_forget, self.tv1)
        ]).mean()

        # Flatten task vectors for vector-level losses
        retain_flat = torch.cat([v.reshape(-1) for v in tv_retain])
        forget_flat = torch.cat([v.reshape(-1) for v in tv_forget])

        # L_orth: squared cosine similarity — drives TV_retain ⊥ TV_forget
        cos_sim = F.cosine_similarity(
            retain_flat.unsqueeze(0), forget_flat.unsqueeze(0)
        )
        l_orth = cos_sim.pow(2)

        # L_norm: scale-invariant penalty — (||TV_retain|| / ||TV1|| - 1)^2
        retain_norm = retain_flat.norm()
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
