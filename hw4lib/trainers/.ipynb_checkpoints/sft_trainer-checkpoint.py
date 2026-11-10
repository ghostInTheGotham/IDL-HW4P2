"""Response-only supervised fine-tuning for the P1 decoder."""
import json
import math
import yaml
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer
from ..decoding.sequence_generator import SequenceGenerator
from ..utils.webnlg import format_webnlg_prompt, score_webnlg


class SFTTrainer(BaseTrainer):
    """Train a decoder on prompt/response pairs.

    Complete the marked training, validation, and generation methods.
    Reuse your P1 model and decoding implementation.
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.best_metric = -float("inf")  # Higher Kaggle score is better.
        self.amp_enabled = bool(config["training"].get("amp", True)) and torch.device(self.device).type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        # Use summed cross-entropy, ignoring labels equal to -100.
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    def _init_experiment(self, run_name, config_file):
        """Provided: use a new output directory, or explicitly resume an existing run."""
        from time import time_ns
        root = (
            Path(self.config.get("output_dir", "expts"))
            .expanduser()
            .resolve()
            / run_name
        )
        resume = self.config["training"].get("resume", False)
        if resume:
            last_checkpoint = root / "checkpoints" / "last.pth"
            if not last_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Cannot resume: {last_checkpoint} was not found. "
                    "If training stopped before the first checkpoint, set this run's resume flag to false and rerun."
                )
        else:
            if root.exists():
                # Preserve any experiment containing saved model weights.
                has_checkpoint = any(
                    path.is_file()
                    and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
                    for path in root.rglob("*")
                )
                if has_checkpoint:
                    raise FileExistsError(
                        f"Experiment {run_name!r} already contains saved model weights. "
                        "Set resume to true to continue from last.pth, or choose a new run name."
                    )
                # Finish the notebook's previous W&B session before
                # moving its files.
                if self.use_wandb:
                    import wandb
                    if wandb.run is not None:
                        wandb.finish()
                archived_root = root.with_name(
                    f"{root.name}-before-first-checkpoint-{time_ns()}"
                )
                root.rename(archived_root)
                print(
                    "Note: The previous attempt has no saved checkpoint. "
                    f"Its files were preserved at {archived_root}. "
                    "Starting this experiment again from the P1 checkpoint."
                )
            root.mkdir(parents=True, exist_ok=False)
            (root / "config.yaml").write_text(yaml.safe_dump(self.config))
            (root / "model.txt").write_text(str(self.model) + "\n")
        checkpoint, attn, text = (root / part for part in ("checkpoints", "attn", "text"))
        for directory in (checkpoint, attn, text):
            directory.mkdir(exist_ok=True)
        if self.use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            self.wandb_run = wandb.init(
                project=self.config["training"].get("wandb_project", "hw4p2"),
                entity=self.config["training"].get("wandb_entity") or None,
                name=run_name, config=self.config,
                id=self.config["training"].get("wandb_run_id") or None,
                resume="must" if resume else None, dir=str(root))
        return root, checkpoint, attn, text, checkpoint / "best.pth", checkpoint / "last.pth"

    @staticmethod
    def _metrics(loss_sum, token_count):
        """Provided: perplexity here is response-token PPL, not character PPL."""
        if token_count <= 0:
            raise ValueError("No supervised response tokens")
        ce = loss_sum / token_count
        if not math.isfinite(ce):
            raise FloatingPointError("Non-finite response loss")
        return {"response_ce": ce, "response_ppl": math.exp(ce), "response_tokens": token_count}

    def _batch(self, batch):
        """Provided: move tensors and reject a batch without supervised targets."""
        inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
        if inputs.ndim != 2 or inputs.shape != labels.shape:
            raise ValueError("input_ids and labels must have matching (B, T) shapes")
        count = int((labels != -100).sum())
        if count == 0:
            raise ValueError("Batch has no supervised response tokens")
        return inputs, labels, count

    def _train_epoch(self, dataloader):
        """Accumulate summed losses; normalize gradients by the cycle's token count.

        Handle a partial final accumulation cycle. Unscale before clipping.
        Advance the scheduler only when the optimizer step ran.
        """
        self._bind_model_to_active_wandb()
        self.model.train()

        train_cfg = self.config.get("training", {})
        grad_accum_steps = int(
            train_cfg.get("gradient_accumulation_steps", train_cfg.get("grad_accum_steps", 1))
        )
        if grad_accum_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        max_norm = train_cfg.get("gradient_clip", train_cfg.get("max_grad_norm", None))

        total_loss_sum = 0.0
        total_tokens = 0
        cycle_tokens = 0

        self.optimizer.zero_grad(set_to_none=True)

        batches = list(dataloader)
        num_batches = len(batches)

        for batch_idx, batch in enumerate(batches):
            inputs, labels, count = self._batch(batch)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                outputs = self.model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss_sum = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            self.scaler.scale(loss_sum).backward()

            loss_val = float(loss_sum.detach().item())
            total_loss_sum += loss_val
            total_tokens += count
            cycle_tokens += count

            is_cycle_end = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == num_batches)
            if is_cycle_end:
                self.scaler.unscale_(self.optimizer)

                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.div_(cycle_tokens)

                if max_norm is not None and float(max_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))

                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                step_ran = (not self.scaler.is_enabled()) or (self.scaler.get_scale() >= scale_before)

                self.optimizer.zero_grad(set_to_none=True)

                if step_ran:
                    self.global_optimizer_step += 1
                    if self.scheduler is not None:
                        self.scheduler.step()

                cycle_tokens = 0

        return self._metrics(total_loss_sum, total_tokens)

    @torch.inference_mode()
    def _validate_epoch(self, dataloader):
        """Compute response CE from total NLL / total supervised tokens."""
        self.model.eval()
        total_loss_sum = 0.0
        total_tokens = 0

        for batch in dataloader:
            inputs, labels, count = self._batch(batch)
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                outputs = self.model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss_sum = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
            total_loss_sum += float(loss_sum.detach().item())
            total_tokens += count

        return self._metrics(total_loss_sum, total_tokens)

    @torch.inference_mode()
    def generate(self, records):
        """Generate from the triples alone, using your P1 greedy decoder.

        Do not append references to prompts. Decode one unpadded prompt at a time.
        Limit total context and stop at generated EOS, excluding prompt text.
        """
        self.model.eval()
        gen_cfg = self.config.get("generation", {})
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 128))
        repeat_penalty = float(gen_cfg.get("repeat_penalty", 1.0))

        model_max_len = getattr(self.model, "max_len", None)
        cfg_max_len = self.config.get("data", {}).get("max_length", None)
        context_limit = model_max_len or cfg_max_len

        score_fn = self.model.score
        cache_score_fn = getattr(self.model, "score_cached", None)

        predictions = []
        for record in records:
            prompt_text = format_webnlg_prompt(record["triples"])
            try:
                prompt_ids = list(self.tokenizer.encode(prompt_text, add_special_tokens=False))
            except TypeError:
                prompt_ids = list(self.tokenizer.encode(prompt_text))

            bos_id = getattr(self.tokenizer, "bos_id", None)
            eos_id = getattr(self.tokenizer, "eos_id", None)
            if eos_id is not None and len(prompt_ids) > 1 and prompt_ids[-1] == eos_id:
                prompt_ids = prompt_ids[:-1]
            if bos_id is not None and (not prompt_ids or prompt_ids[0] != bos_id):
                prompt_ids = [bos_id] + prompt_ids

            if context_limit is not None:
                max_prompt_len = max(1, int(context_limit) - max_new_tokens)
                if len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[:max_prompt_len]
                total_max_length = min(int(context_limit), len(prompt_ids) + max_new_tokens)
            else:
                total_max_length = len(prompt_ids) + max_new_tokens

            prompt_len = len(prompt_ids)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            generator = SequenceGenerator(
                score_fn=score_fn,
                tokenizer=self.tokenizer,
                max_length=total_max_length,
                device=str(self.device),
                cache_score_fn=cache_score_fn,
            )
            sequences, _ = generator.generate_greedy(prompt_tensor, repeat_penalty=repeat_penalty)
            processed = SequenceGenerator.post_process_sequence(
                sequences[0], self.tokenizer, start_index=prompt_len
            )

            response_ids = processed[prompt_len:].tolist()
            if eos_id is not None and response_ids and response_ids[-1] == eos_id:
                response_ids = response_ids[:-1]

            try:
                pred_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            except TypeError:
                pred_text = self.tokenizer.decode(response_ids)
            predictions.append(pred_text.strip() if isinstance(pred_text, str) else pred_text)

        return predictions

    def evaluate(self, dataloader, records=None):
        """Provided: combine teacher-forced response loss with generated-text metrics."""
        metrics = self._validate_epoch(dataloader)
        if records is not None:
            predictions = self.generate(records)
            scores = score_webnlg(predictions, [row["references"] for row in records])
            metrics.update(bleu=scores["bleu"], rouge_l=scores["rouge_l"],
                           kaggle_score=scores["kaggle_score"], examples=len(records))
        return metrics

    def train(self, train_dataloader, val_dataloader, dev_records):
        """Run epochs and select the best generated dev Kaggle-score checkpoint.

        Save last.pth for resuming the next epoch and best.pth for prediction.
        Log token loss and generation scores as different quantities.
        """
        epochs = int(self.config["training"]["epochs"])
        if self.config["training"].get("resume", False) and self.current_epoch == 0 and self.last_checkpoint_path.is_file():
            self.load_checkpoint("last.pth")

        for epoch in range(self.current_epoch, epochs):
            train_metrics = self._train_epoch(train_dataloader)
            val_metrics = self.evaluate(val_dataloader, dev_records)

            self.current_epoch = epoch + 1

            kaggle_score = val_metrics.get("kaggle_score", -float("inf"))
            is_best = kaggle_score > self.best_metric
            if is_best:
                self.best_metric = kaggle_score

            if self.use_wandb:
                import wandb
                log_payload = {
                    "epoch": self.current_epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }
                wandb.log(log_payload, step=self.global_optimizer_step)

            self.save_checkpoint("last.pth")
            if is_best:
                self.save_checkpoint("best.pth")