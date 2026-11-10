"""Tests that scheduler durations count optimizer updates, not microbatches."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from hw4lib.utils.create_lr_scheduler import _epochs_to_steps, create_scheduler


class TestAccumulationAwareScheduler(unittest.TestCase):
    def setUp(self):
        """Create a five-microbatch DataLoader."""
        self.loader = DataLoader(
            TensorDataset(torch.arange(5)),
            batch_size=1,
            shuffle=False,
        )

    def test_epoch_conversion_includes_partial_accumulation_cycle(self):
        """Count a final partial accumulation cycle as one update."""
        self.assertEqual(_epochs_to_steps(2, self.loader, 2), 6)
        self.assertEqual(_epochs_to_steps(1, self.loader, 10), 1)

    def test_cosine_and_warmup_use_optimizer_step_counts(self):
        """Configure cosine and warmup durations in optimizer updates."""
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        config = {
            "name": "cosine",
            "cosine": {"T_max": 2, "eta_min": 0.0},
            "warmup": {
                "enabled": True,
                "epochs": 1,
                "start_factor": 0.1,
                "end_factor": 1.0,
            },
        }

        scheduler = create_scheduler(
            optimizer,
            config,
            self.loader,
            gradient_accumulation_steps=2,
        )

        self.assertEqual(scheduler._milestones, [3])
        self.assertEqual(scheduler._schedulers[0].total_iters, 3)
        self.assertEqual(scheduler._schedulers[1].T_max, 6)

    def test_invalid_accumulation_is_rejected(self):
        """Reject nonpositive accumulation settings."""
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _epochs_to_steps(1, self.loader, 0)


if __name__ == "__main__":
    unittest.main()
