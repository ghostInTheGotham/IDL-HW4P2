"""Smoke test for LMTrainer with fixed, unpadded GPT-2 chunks."""

import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hw4lib.data import GPT2Tokenizer, LMDataset, write_token_shards
from hw4lib.model import DecoderOnlyTransformer
from hw4lib.trainers import LMTrainer


def test_lm_trainer_fixed_chunk_epoch():
    """Exercise fixed-chunk training, validation, generation, and resume."""
    original_directory = Path.cwd()
    original_output_root = os.environ.get("HW4P1_OUTPUT_ROOT")
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        try:
            os.chdir(work_dir)
            output_root = (work_dir / "persistent-runs").resolve()
            os.environ["HW4P1_OUTPUT_ROOT"] = str(output_root)
            tokenizer = GPT2Tokenizer()
            write_token_shards(
                ["small language-model training example " * 20],
                work_dir / "data" / "train",
                tokenizer,
                shard_size=256,
            )
            dataset = LMDataset(
                work_dir / "data",
                "train",
                context_length=4,
            )
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

            model = DecoderOnlyTransformer(
                num_layers=1,
                d_model=8,
                num_heads=2,
                num_kv_heads=1,
                d_ff=16,
                dropout=0.0,
                max_len=8,
                num_classes=tokenizer.vocab_size,
                weight_tying=True,
            )
            config = {
                "data": {"batch_size": 2},
                "training": {
                    "gradient_accumulation_steps": 4,
                    "label_smoothing": 0.0,
                    "use_wandb": False,
                },
            }
            config_path = work_dir / "config.yaml"
            config_path.write_text("training: smoke-test\n", encoding="utf-8")

            trainer = LMTrainer(
                model=model,
                tokenizer=tokenizer,
                config=config,
                run_name="smoke",
                config_file=str(config_path),
                device="cpu",
            )
            assert trainer.expt_root == output_root / "smoke"
            trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
                trainer.optimizer,
                lr_lambda=lambda _step: 1.0,
            )

            train_metrics, train_attention = trainer._train_epoch(dataloader)
            val_metrics, val_attention = trainer._validate_epoch(dataloader)
            generated = trainer.generate(
                dataloader,
                generation_config={
                    "num_samples": 1,
                    "prompt_length": 2,
                    "seed": 7,
                    "max_length": 4,
                    "beam_width": 1,
                    "repeat_penalty": 1.0,
                    "top_k": 0,
                    "top_p": 0.0,
                    "use_cache": True,
                },
            )
            generated_with_derived_prompt = trainer.generate(
                dataloader,
                generation_config={
                    "num_samples": 1,
                    "seed": 7,
                    "max_length": 4,
                    "beam_width": 1,
                    "repeat_penalty": 1.0,
                    "top_k": 0,
                    "top_p": 0.0,
                    "use_cache": True,
                },
            )
            generation_configs = trainer._get_evaluation_generation_configs(
                dataloader
            )

            assert train_metrics["perplexity_token"] > 0
            assert val_metrics["perplexity_char"] > 0
            assert "layer1_dec_self" in train_attention
            assert "layer1_dec_self" in val_attention
            assert train_attention["layer1_dec_self"].shape[0] == 1
            assert val_attention["layer1_dec_self"].shape[0] == 1
            assert trainer.scheduler.last_epoch == 4
            assert trainer.global_optimizer_step == 4
            assert len(generated) == 1
            assert len(generated_with_derived_prompt) == 1
            assert set(generated[0]) == {"prompt", "original", "generated", "score"}
            assert set(generation_configs) == {"greedy", "sample"}
            assert "temperature" not in generation_configs["greedy"]
            assert generation_configs["sample"]["temperature"] == 1.0

            additive_model = DecoderOnlyTransformer(
                num_layers=1,
                d_model=8,
                num_heads=2,
                d_ff=16,
                dropout=0.0,
                max_len=4,
                num_classes=tokenizer.vocab_size,
                use_rope=False,
            )
            additive_trainer = LMTrainer(
                model=additive_model,
                tokenizer=tokenizer,
                config=config,
                run_name="additive-smoke",
                config_file=str(config_path),
                device="cpu",
            )
            additive_generated = additive_trainer.generate(
                dataloader,
                generation_config={
                    "num_samples": 1,
                    "prompt_length": 2,
                    "seed": 7,
                    "max_length": 4,
                    "beam_width": 1,
                    "repeat_penalty": 1.0,
                    "top_k": 0,
                    "top_p": 0.0,
                },
            )
            assert len(additive_generated) == 1
            try:
                additive_trainer.generate(
                    dataloader,
                    generation_config={
                        "num_samples": 1,
                        "prompt_length": 2,
                        "max_length": 4,
                        "use_cache": True,
                    },
                )
            except ValueError as exc:
                assert "rotary decoder" in str(exc)
            else:
                raise AssertionError("Additive decoders must reject KV caching")

            try:
                trainer.generate(
                    dataloader,
                    generation_config={
                        "num_samples": 1,
                        "prompt_length": dataset.context_length,
                        "max_length": model.max_len,
                    },
                )
            except ValueError as exc:
                assert "dataset context_length" in str(exc)
            else:
                raise AssertionError("Invalid prompt length should fail before sampling")

            try:
                trainer.generate(
                    dataloader,
                    generation_config={
                        "num_samples": 1,
                        "prompt_length": 2,
                        "max_length": 4,
                        "beam_width": 2,
                        "top_k": 5,
                    },
                )
            except ValueError as exc:
                assert "cannot be combined" in str(exc)
            else:
                raise AssertionError("Beam search and sampling controls must conflict")

            def fixed_train_epoch(_dataloader):
                """Return deterministic training metrics for resume testing.

                Args:
                    _dataloader: Unused training DataLoader.
                """
                metrics = {"ce_loss_char": 0.4}
                result = (metrics, {})
                return result

            def fixed_validate_epoch(_dataloader):
                """Return a worse validation result for resume testing.

                Args:
                    _dataloader: Unused validation DataLoader.
                """
                metrics = {"ce_loss_char": 0.6}
                result = (metrics, {})
                return result

            def no_generation(_dataloader):
                """Skip generation during the checkpoint-state test.

                Args:
                    _dataloader: Unused generation DataLoader.
                """
                results = []
                return results

            trainer._train_epoch = fixed_train_epoch
            trainer._validate_epoch = fixed_validate_epoch
            trainer.generate = no_generation
            trainer._log_metrics = lambda *_args: None
            trainer._save_generated_text = lambda *_args: None
            trainer.current_epoch = 3
            trainer.best_metric = 0.5
            trainer.train(dataloader, dataloader, epochs=1)

            checkpoint_path = (
                trainer.checkpoint_dir / "checkpoint-last-epoch-model.pth"
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            assert checkpoint["epoch"] == 4
            assert checkpoint["global_optimizer_step"] == 4
            assert checkpoint["best_metric"] == 0.5
            assert set(checkpoint) == {
                "submission_tracking",
                "format_version",
                "epoch",
                "global_optimizer_step",
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "scaler_state_dict",
                "best_metric",
                "training_history",
                "config",
                "rng_state",
            }

            expected_python_random = random.random()
            expected_numpy_random = np.random.random()
            expected_torch_random = torch.rand(3)
            random.random()
            np.random.random()
            torch.rand(3)

            trainer.current_epoch = 0
            trainer.global_optimizer_step = 0
            trainer.best_metric = float("inf")
            trainer.load_checkpoint("checkpoint-last-epoch-model.pth")
            assert trainer.current_epoch == 4
            assert trainer.global_optimizer_step == 4
            assert trainer.best_metric == 0.5
            assert random.random() == expected_python_random
            assert np.random.random() == expected_numpy_random
            assert torch.equal(torch.rand(3), expected_torch_random)

            checkpoint["legacy_metadata"] = "ignored"
            legacy_checkpoint = trainer.checkpoint_dir / "legacy-checkpoint.pth"
            torch.save(checkpoint, legacy_checkpoint)
            trainer.load_checkpoint(legacy_checkpoint.name)
            assert trainer.current_epoch == 4

        finally:
            os.chdir(original_directory)
            if original_output_root is None:
                os.environ.pop("HW4P1_OUTPUT_ROOT", None)
            else:
                os.environ["HW4P1_OUTPUT_ROOT"] = original_output_root


def test_uneven_accumulation_matches_one_large_batch():
    """A short final microbatch must receive the correct token weight."""
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        try:
            os.chdir(work_dir)
            tokenizer = GPT2Tokenizer()
            assert LMTrainer._optimizer_step_ran(1024.0, 1024.0)
            assert LMTrainer._optimizer_step_ran(1024.0, 2048.0)
            assert not LMTrainer._optimizer_step_ran(1024.0, 512.0)

            class ThreeExampleDataset(Dataset):
                """Three fixed examples for an uneven two-microbatch cycle."""

                vocab_size = tokenizer.vocab_size
                eos_id = tokenizer.eos_id

                def __len__(self) -> int:
                    """Get the number of examples.

                    Returns:
                        Number of examples.
                    """
                    length = 3
                    return length

                def __getitem__(self, idx: int):
                    """Get one shifted token pair.

                    Args:
                        idx: Example index.

                    Returns:
                        Input and target token tensors.
                    """
                    start = idx + 1
                    tokens = torch.arange(start, start + 5, dtype=torch.long)
                    sample = (tokens[:-1], tokens[1:])
                    return sample

                def get_avg_chars_per_token(self) -> float:
                    """Get the test corpus character ratio.

                    Returns:
                        One character per token.
                    """
                    ratio = 1.0
                    return ratio

            def make_model() -> DecoderOnlyTransformer:
                """Build the small model used by both update paths."""
                model = DecoderOnlyTransformer(
                    num_layers=1,
                    d_model=8,
                    num_heads=2,
                    num_kv_heads=1,
                    d_ff=16,
                    dropout=0.0,
                    max_len=4,
                    num_classes=tokenizer.vocab_size,
                )
                return model

            torch.manual_seed(13)
            accumulated_model = make_model()
            large_batch_model = make_model()
            large_batch_model.load_state_dict(accumulated_model.state_dict())

            config_path = work_dir / "config.yaml"
            config_path.write_text("training: accumulation-test\n", encoding="utf-8")

            def make_trainer(model, run_name, accumulation_steps):
                """Build a trainer with one deterministic SGD update.

                Args:
                    model: Decoder-only model to train.
                    run_name: Temporary experiment name.
                    accumulation_steps: Microbatches per update.

                Returns:
                    Configured LM trainer.
                """
                config = {
                    "data": {"batch_size": 3},
                    "training": {
                        "gradient_accumulation_steps": accumulation_steps,
                        "label_smoothing": 0.0,
                        "use_wandb": False,
                    },
                }
                trainer = LMTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    run_name=run_name,
                    config_file=str(config_path),
                    device="cpu",
                )
                trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    trainer.optimizer,
                    lr_lambda=lambda _step: 1.0,
                )
                return trainer

            dataset = ThreeExampleDataset()
            accumulated_loader = DataLoader(dataset, batch_size=2, shuffle=False)
            large_batch_loader = DataLoader(dataset, batch_size=3, shuffle=False)
            accumulated_trainer = make_trainer(
                accumulated_model,
                "accumulated",
                accumulation_steps=2,
            )
            large_batch_trainer = make_trainer(
                large_batch_model,
                "large-batch",
                accumulation_steps=1,
            )

            accumulated_metrics, _ = accumulated_trainer._train_epoch(
                accumulated_loader
            )
            large_batch_metrics, _ = large_batch_trainer._train_epoch(
                large_batch_loader
            )

            assert accumulated_trainer.global_optimizer_step == 1
            assert large_batch_trainer.global_optimizer_step == 1

            assert abs(
                accumulated_metrics["ce_loss_token"]
                - large_batch_metrics["ce_loss_token"]
            ) < 1e-5
            for name, accumulated_parameter in accumulated_model.state_dict().items():
                large_batch_parameter = large_batch_model.state_dict()[name]
                assert torch.allclose(
                    accumulated_parameter,
                    large_batch_parameter,
                    atol=1e-6,
                    rtol=1e-6,
                )

        finally:
            os.chdir(original_directory)


if __name__ == "__main__":
    test_lm_trainer_fixed_chunk_epoch()
    test_uneven_accumulation_matches_one_large_batch()
    print("LMTrainer fixed-chunk smoke test passed.")
