from bisect import bisect_right
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SHARD_DTYPE = np.dtype("<u2")
MANIFEST_FILENAME = "manifest.json"
FORMAT_VERSION = 3


class LMDataset(Dataset):
    """
    Dataset for fixed-window Language Model training/evaluation.

    Each item comes from ``context_length + 1`` contiguous token IDs::

        tokens = shard[start : start + context_length + 1]
        inputs = tokens[:-1]
        targets = tokens[1:]

    Shards are memory-mapped and samples never cross a shard boundary. Since
    every item has the same shape, a regular PyTorch ``DataLoader`` can collate
    batches without a custom ``collate_fn``.
    """

    def __init__(
        self,
        root: str | Path,
        partition: str,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        """Initialize a partition of token shards."""
        super().__init__()

        # Initialize the cache early so failed validation in __init__ never leaves __del__ broken
        self._mmap_cache: dict[int, np.memmap] = {}

        # Validate the context length
        if (
            not isinstance(context_length, Integral)
            or isinstance(context_length, bool)
            or context_length <= 0
        ):
            raise ValueError("context_length must be a positive integer")

        # Use non-overlapping input chunks when no stride is provided
        if stride is None:
            stride = int(context_length)

        # Validate the stride
        if (
            not isinstance(stride, Integral)
            or isinstance(stride, bool)
            or stride <= 0
        ):
            raise ValueError("stride must be a positive integer")

        # Store configuration and other args
        self.root = Path(root).resolve()
        self.partition = partition
        self.context_length = int(context_length)
        self.stride = int(stride)

        # Set up data paths
        self.data_dir = (self.root / partition).resolve()
        self.manifest_path = self.data_dir / MANIFEST_FILENAME

        # Load the partition manifest
        manifest = self._load_manifest(self.manifest_path)

        # Store tokenizer and corpus statistics
        self.vocab_size = int(manifest["vocab_size"])
        self.eos_id = int(manifest["eos_id"])
        self.total_chars = int(manifest["num_chars"])
        self.total_tokens = int(manifest["num_tokens"])
        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens else 0.0
        )

        # Initialize lists that describe the token shards and their windows
        self.shard_paths: list[Path] = []
        self.shard_lengths: list[int] = []
        self.windows_per_shard: list[int] = []
        self.cumulative_windows: list[int] = []

        # Initialize tracking variables
        cumulative_windows = 0
        window_size = self.context_length + 1

        # Count valid fixed windows in each shard
        for shard in manifest["shards"]:
            shard_path, num_tokens = self._shard_entry(shard)
            self.shard_paths.append(shard_path)
            self.shard_lengths.append(num_tokens)

            if num_tokens >= window_size:
                num_windows = (num_tokens - window_size) // self.stride + 1
            else:
                num_windows = 0

            self.windows_per_shard.append(num_windows)
            cumulative_windows += num_windows
            self.cumulative_windows.append(cumulative_windows)

        self.length = cumulative_windows

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        """Read the partition's shard list and corpus statistics."""
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def _shard_entry(self, shard: dict) -> tuple[Path, int]:
        return self.data_dir / shard["file"], int(shard["num_tokens"])

    def __len__(self) -> int:
        return self.length

    def _get_shard(self, shard_index: int) -> np.memmap:
        """Open a shard lazily and reuse its memory map."""
        if shard_index not in self._mmap_cache:
            shard_path = self.shard_paths[shard_index]
            self._mmap_cache[shard_index] = np.memmap(
                shard_path,
                dtype=SHARD_DTYPE,
                mode="r",
            )
        return self._mmap_cache[shard_index]

    def close(self) -> None:
        """Close open memory maps so temporary files can be cleaned up on Windows."""
        cache = getattr(self, "_mmap_cache", None)
        if cache is not None:
            for mmap_obj in list(cache.values()):
                try:
                    if hasattr(mmap_obj, "_mmap") and mmap_obj._mmap is not None:
                        mmap_obj._mmap.close()
                except Exception:
                    pass
            cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one next-token training example."""
        if not isinstance(idx, Integral):
            raise TypeError("Dataset index must be an integer")
        index = int(idx)
        if index < 0 or index >= self.length:
            raise IndexError("Dataset index out of range")

        shard_index = bisect_right(self.cumulative_windows, index)
        previous_windows = self.cumulative_windows[shard_index - 1] if shard_index > 0 else 0
        local_index = index - previous_windows

        start = local_index * self.stride
        stop = start + self.context_length + 1

        shard = self._get_shard(shard_index)
        window_array = np.array(shard[start:stop], dtype=np.int64)

        window = torch.from_numpy(window_array).long()
        inputs = window[:-1]
        targets = window[1:]

        return inputs, targets

    def collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack fixed-length examples into a batch."""
        inputs, targets = zip(*batch)
        batched_inputs = torch.stack(inputs, dim=0)
        batched_targets = torch.stack(targets, dim=0)
        return batched_inputs, batched_targets

    def get_avg_chars_per_token(self) -> float:
        return self.avg_chars_per_token

    def sample_prompts(
        self,
        num_samples: int,
        prompt_length: int,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample fixed prompts and their source chunks."""
        if not isinstance(num_samples, int) or isinstance(num_samples, bool):
            raise ValueError("num_samples must be a positive integer")
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if not isinstance(prompt_length, int) or isinstance(prompt_length, bool):
            raise ValueError("prompt_length must be a positive integer")
        if not 0 < prompt_length < self.context_length:
            raise ValueError("prompt_length must be between 1 and context_length - 1")

        generator = np.random.default_rng(seed)
        indices = generator.integers(0, len(self), size=num_samples)
        originals = [self[int(index)][0] for index in indices]
        prompts = torch.stack(
            [original[:prompt_length].clone() for original in originals]
        )
        return prompts, originals

    def __getstate__(self) -> dict[str, Any]:
        """Prepare dataset state for a DataLoader worker."""
        state = self.__dict__.copy()
        state["_mmap_cache"] = {}
        return state