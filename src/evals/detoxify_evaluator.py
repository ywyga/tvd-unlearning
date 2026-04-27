import json
import logging
import os

import torch

from evals.base import Evaluator

logger = logging.getLogger("evaluator")


class DetoxifyEvaluator(Evaluator):
    """Evaluates toxicity of model generations using the Detoxify library.

    Loads toxic prompts from a JSONL file, generates continuations with the model,
    scores each continuation with Detoxify, and reports mean toxicity score and
    toxicity rate (fraction of outputs above a configurable threshold).
    """

    def __init__(self, eval_cfg, **kwargs):
        self.name = "Detoxify"
        self.eval_cfg = eval_cfg

    def prepare_model(self, model):
        model.eval()
        return model

    def _load_prompts(self, data_files, num_prompts, prefix_words):
        """Load up to num_prompts prompts from a JSONL file, truncated to prefix_words words."""
        prompts = []
        with open(data_files, "r", encoding="utf-8") as f:
            for line in f:
                if len(prompts) >= num_prompts:
                    break
                row = json.loads(line)
                text = row.get("text", "").strip()
                if not text:
                    continue
                words = text.split()
                prompt = " ".join(words[:prefix_words])
                if prompt:
                    prompts.append(prompt)
        return prompts

    def _generate_continuations(self, model, tokenizer, prompts, max_new_tokens, batch_size):
        """Batch-generate continuations for a list of prompts."""
        device = next(model.parameters()).device
        continuations = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Decode only the newly generated tokens (strip the prompt)
            input_len = inputs["input_ids"].shape[1]
            for ids in output_ids:
                new_tokens = ids[input_len:]
                continuations.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

        return continuations

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        try:
            from detoxify import Detoxify
        except ImportError:
            raise ImportError(
                "detoxify is required for DetoxifyEvaluator. "
                "Install it with: pip install 'open-unlearning[detoxify]'"
            )

        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

        if not overwrite and os.path.exists(summary_file_path):
            logger.info(f"Loading existing Detoxify results from {summary_file_path}")
            return self.load_logs_from_file(summary_file_path)

        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None:
            raise ValueError("DetoxifyEvaluator requires a tokenizer")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = self.prepare_model(model)

        cfg = self.eval_cfg
        logger.info("***** Running Detoxify evaluation suite *****")
        logger.info(f"Loading prompts from {cfg.prompt_data_files}")

        prompts = self._load_prompts(
            cfg.prompt_data_files,
            cfg.num_prompts,
            cfg.prompt_prefix_words,
        )
        logger.info(f"Generating continuations for {len(prompts)} prompts")

        continuations = self._generate_continuations(
            model, tokenizer, prompts, cfg.max_new_tokens, cfg.batch_size
        )

        logger.info(f"Scoring {len(continuations)} continuations with Detoxify ({cfg.detoxify_model})")
        scorer = Detoxify(cfg.detoxify_model)
        scores = scorer.predict(continuations)
        toxicity_scores = scores["toxicity"]

        mean_score = float(sum(toxicity_scores) / len(toxicity_scores))
        toxicity_rate = float(
            sum(1 for s in toxicity_scores if s >= cfg.toxicity_threshold) / len(toxicity_scores)
        )

        per_sample_logs = [
            {"prompt": p, "continuation": c, "toxicity": float(s)}
            for p, c, s in zip(prompts, continuations, toxicity_scores)
        ]
        self.save_logs({"samples": per_sample_logs}, logs_file_path)

        summary = {
            "detoxify/mean_toxicity": mean_score,
            "detoxify/toxicity_rate": toxicity_rate,
            "detoxify/num_prompts": len(prompts),
        }
        logger.info(f"Detoxify mean_toxicity: {mean_score:.4f}  toxicity_rate: {toxicity_rate:.4f}")
        self.save_logs(summary, summary_file_path)
        return summary
