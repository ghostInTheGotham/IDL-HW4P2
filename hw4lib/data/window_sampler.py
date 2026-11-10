"""Seeded training-window sampling for the notebook."""

import random
from torch.utils.data import Sampler


class EpochWindowSampler(Sampler[int]):
    """Sample without replacement; None uses the whole training partition.

    Call set_epoch before training each epoch. Previewing the loader does not
    change its order, and an epoch-boundary resume uses the same window indices.
    The original dataset stays on the DataLoader, including its corpus metadata.
    """

    def __init__(self, dataset, max_windows: int | None = None, seed: int = 11785):
        self.dataset_size = len(dataset)
        if self.dataset_size <= 0:
            raise ValueError("Training dataset must contain at least one window")
        if max_windows is not None and (
            isinstance(max_windows, bool)
            or not isinstance(max_windows, int)
            or max_windows <= 0
        ):
            raise ValueError("max_windows must be a positive integer or None")
        self.num_samples = (self.dataset_size if max_windows is None
                            else min(max_windows, self.dataset_size))
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        return iter(rng.sample(range(self.dataset_size), self.num_samples))
