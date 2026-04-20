import torch
from transformers import AutoModelForCausalLM
from transformers.trainer_utils import TrainOutput
from trainer.unlearn.base import UnlearnTrainer


class TaskArithmetic(UnlearnTrainer):
    """
    Task Arithmetic unlearning (Ilharco et al., ICLR 2023 — negation variant).

    Computes the forget task vector and subtracts it from the target model:

        TV_forget = M_forget - M0
        M_unlearned = M1 - scale * TV_forget

    where:
      M1         = the starting (fully fine-tuned) model provided to the trainer
      M0         = frozen base / pre-trained model (base_model_name_or_path)
      M_forget   = a model fine-tuned *only* on the forget split (forget_model_path)
      scale      = scalar controlling negation strength (default 1.0)

    This is a closed-form, training-free operation.  The `train()` method applies
    the arithmetic once and returns immediately; no gradient computation is done.

    Prerequisites
    -------------
    M_forget must be fine-tuned in advance on the forget split, e.g.:

        python src/train.py experiment=finetune/tofu/default.yaml \\
            task_name=tofu_phi-1_5_forget10 \\
            model=phi-1_5 \\
            data/datasets@data.train=TOFU_QA_forget \\
            data.train.TOFU_QA_forget.args.hf_args.name=forget10
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        forget_model_path: str,
        scale: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.forget_model_path = forget_model_path
        self.scale = scale

    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        """Apply task-arithmetic negation in-place and return without training."""
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # Load M0 and M_forget onto CPU — only need them for arithmetic, not inference
        m0 = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path, torch_dtype=dtype
        )
        m_forget = AutoModelForCausalLM.from_pretrained(
            self.forget_model_path, torch_dtype=dtype
        )

        # M_unlearned = M1 - scale * (M_forget - M0)
        with torch.no_grad():
            for p_result, p_forget, p_base in zip(
                self.model.parameters(),
                m_forget.parameters(),
                m0.parameters(),
            ):
                tv_forget = p_forget.data.to(device) - p_base.data.to(device)
                p_result.data -= self.scale * tv_forget

        # Free GPU memory used by the loaded models
        del m0, m_forget

        return TrainOutput(global_step=0, training_loss=0.0, metrics={})
