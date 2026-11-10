import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader

from hw4lib.data import GPT2Tokenizer, H4Tokenizer
from hw4lib.data.lm_dataset import FORMAT_VERSION, LMDataset, SHARD_DTYPE
from hw4lib.data.token_shards import write_token_shards


EOS_TOKEN = "<|endoftext|>"
CANONICAL_TOKENIZER_PATH = (
    Path(__file__).parents[1]
    / "hw4lib"
    / "data"
    / "tokenizer_jsons"
    / "tokenizer_gpt2.json"
)


def _build_tokenizer(path: Path) -> GPT2Tokenizer:
    """Build a small byte-level BPE tokenizer for offline wrapper tests.

    Args:
        path: Output tokenizer JSON path.

    Returns:
        Wrapped byte-level tokenizer.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        ["Hello, world!", "naive cafe", "line one\nline two", "GPT-2 BPE"],
        trainer=trainer,
    )
    tokenizer.save(str(path))
    wrapped_tokenizer = GPT2Tokenizer(path)
    return wrapped_tokenizer


def _write_partition(
    root: Path,
    shards: list[tuple[str, list[int]]],
    *,
    partition: str = "train",
    num_chars: int = 30,
    vocab_size: int = 50257,
    eos_id: int = 50256,
) -> Path:
    """Write a small manifest and its raw token shards.

    Args:
        root: Temporary dataset root.
        shards: Filename and token-ID pairs.
        partition: Partition directory name.
        num_chars: Source character count stored in the manifest.
        vocab_size: Manifest vocabulary size.
        eos_id: Manifest end-token ID.

    Returns:
        Written partition directory.
    """
    data_dir = root / partition
    data_dir.mkdir(parents=True)
    entries = []
    for filename, token_ids in shards:
        shard_path = data_dir / filename
        np.asarray(token_ids, dtype=SHARD_DTYPE).tofile(shard_path)
        entries.append(
            {
                "file": filename,
                "num_tokens": len(token_ids),
            }
        )

    manifest = {
        "format_version": FORMAT_VERSION,
        "dtype": SHARD_DTYPE.str,
        "tokenizer": "gpt2",
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "num_chars": num_chars,
        "num_tokens": sum(len(tokens) for _, tokens in shards),
        "shards": entries,
    }
    with (data_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file)
    return data_dir


class TestGPT2Tokenizer(unittest.TestCase):
    def setUp(self):
        """Create a temporary byte-level tokenizer."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.tokenizer_path = self.root / "tokenizer.json"
        self.tokenizer = _build_tokenizer(self.tokenizer_path)

    def tearDown(self):
        """Remove temporary tokenizer files."""
        self.temp_dir.cleanup()

    def test_round_trip_and_no_implicit_eos(self):
        """Check byte-level round trips without automatic EOT insertion."""
        text = "Hello, café!\n🙂"
        token_ids = self.tokenizer.encode(text)

        self.assertNotIn(self.tokenizer.eos_id, token_ids)
        self.assertEqual(self.tokenizer.decode(token_ids), text)
        self.assertTrue(self.tokenizer.tokenize(text))

    def test_h4_tokenizer_supports_gpt2(self):
        """Check GPT-2 through the shared H4Tokenizer interface."""
        tokenizer = H4Tokenizer(
            token_map={"gpt2": self.tokenizer_path},
            token_type="gpt2",
            validate=False,
        )

        text = "One H4Tokenizer interface."
        token_ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.token_type, "gpt2")
        self.assertEqual(tokenizer.decode(token_ids), text)
        self.assertEqual(tokenizer.eos_id, self.tokenizer.eos_id)
        self.assertIsInstance(self.tokenizer, H4Tokenizer)

    def test_missing_file_is_rejected(self):
        """Reject a missing tokenizer JSON."""
        with self.assertRaisesRegex(FileNotFoundError, "Tokenizer file not found"):
            GPT2Tokenizer(self.root / "missing.json")

    def test_missing_eot_is_rejected(self):
        """Reject a tokenizer without GPT-2's end token."""
        tokenizer = Tokenizer(models.WordLevel({"hello": 0}, unk_token="hello"))
        path = self.root / "no-eot.json"
        tokenizer.save(str(path))

        with self.assertRaisesRegex(ValueError, "missing the GPT-2 end token"):
            GPT2Tokenizer(path)

    @unittest.skipUnless(
        CANONICAL_TOKENIZER_PATH.is_file(),
        "Canonical GPT-2 tokenizer JSON has not been bundled yet",
    )
    def test_canonical_gpt2_ids(self):
        """Check canonical vocabulary and example token IDs."""
        tokenizer = GPT2Tokenizer()
        self.assertEqual(tokenizer.vocab_size, 50257)
        self.assertEqual(tokenizer.eos_id, 50256)
        self.assertEqual(
            tokenizer.encode("Hello, world!"),
            [15496, 11, 995, 0],
        )


class TestLMDataset(unittest.TestCase):
    def setUp(self):
        """Create a temporary dataset root."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        """Remove temporary dataset files."""
        self.temp_dir.cleanup()

    def test_exact_shifted_windows(self):
        """Check exact non-overlapping next-token windows."""
        _write_partition(self.root, [("shard_00000.bin", list(range(13)))])
        dataset = LMDataset(self.root, "train", context_length=4)

        self.assertEqual(len(dataset), 3)
        x0, y0 = dataset[0]
        x2, y2 = dataset[2]
        self.assertEqual(x0.tolist(), [0, 1, 2, 3])
        self.assertEqual(y0.tolist(), [1, 2, 3, 4])
        self.assertEqual(x2.tolist(), [8, 9, 10, 11])
        self.assertEqual(y2.tolist(), [9, 10, 11, 12])
        self.assertEqual(x0.dtype, torch.long)
        self.assertEqual(y0.dtype, torch.long)

    def test_windows_do_not_require_manifest_header_validation(self):
        directory = _write_partition(self.root, [("shard.bin", list(range(9)))])
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        for key in ("format_version", "dtype", "tokenizer"):
            manifest.pop(key)
        path.write_text(json.dumps(manifest))
        dataset = LMDataset(self.root, "train", context_length=4)
        self.assertEqual(dataset[1][0].tolist(), [4, 5, 6, 7])

    def test_stride_and_shard_boundaries(self):
        """Check custom strides without crossing shard boundaries."""
        _write_partition(
            self.root,
            [
                ("shard_00000.bin", list(range(9))),
                ("shard_00001.bin", list(range(100, 109))),
            ],
        )
        dataset = LMDataset(self.root, "train", context_length=4, stride=2)

        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset[2][0].tolist(), [4, 5, 6, 7])
        self.assertEqual(dataset[3][0].tolist(), [100, 101, 102, 103])

    def test_short_shard_is_skipped_and_manifest_order_is_used(self):
        """Skip short shards while preserving manifest order."""
        _write_partition(
            self.root,
            [
                ("z-short.bin", [50, 51]),
                ("b-second.bin", [20, 21, 22, 23, 24]),
                ("a-third.bin", [30, 31, 32, 33, 34]),
            ],
        )
        dataset = LMDataset(self.root, "train", context_length=4)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0][0].tolist(), [20, 21, 22, 23])
        self.assertEqual(dataset[1][0].tolist(), [30, 31, 32, 33])

    def test_default_dataloader_collates_fixed_windows(self):
        """Check PyTorch's default fixed-window collation."""
        _write_partition(self.root, [("shard.bin", list(range(13)))])
        dataset = LMDataset(self.root, "train", context_length=4)

        x, y = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        self.assertEqual(tuple(x.shape), (2, 4))
        self.assertEqual(tuple(y.shape), (2, 4))
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)

    def test_optional_collate_method_matches_default(self):
        """Check the optional collator against PyTorch's default."""
        _write_partition(self.root, [("shard.bin", list(range(13)))])
        dataset = LMDataset(self.root, "train", context_length=4)
        default_batch = next(
            iter(DataLoader(dataset, batch_size=2, shuffle=False))
        )
        explicit_batch = next(
            iter(
                DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    collate_fn=dataset.collate_fn,
                )
            )
        )

        self.assertTrue(torch.equal(default_batch[0], explicit_batch[0]))
        self.assertTrue(torch.equal(default_batch[1], explicit_batch[1]))

    def test_sample_prompts_is_reproducible(self):
        """Check seeded prompt sampling and source chunks."""
        _write_partition(self.root, [("shard.bin", list(range(25)))])
        dataset = LMDataset(self.root, "train", context_length=6)

        prompts_a, originals_a = dataset.sample_prompts(3, prompt_length=2, seed=7)
        prompts_b, originals_b = dataset.sample_prompts(3, prompt_length=2, seed=7)

        self.assertTrue(torch.equal(prompts_a, prompts_b))
        self.assertEqual(tuple(prompts_a.shape), (3, 2))
        self.assertTrue(all(len(original) == 6 for original in originals_a))
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(originals_a, originals_b))
        )

    def test_character_metric_comes_from_manifest(self):
        """Read the corpus character ratio from manifest statistics."""
        _write_partition(
            self.root,
            [("shard.bin", list(range(10)))],
            num_chars=25,
        )
        dataset = LMDataset(self.root, "train", context_length=3)

        self.assertAlmostEqual(dataset.get_avg_chars_per_token(), 2.5)


    def test_mmap_is_lazy_and_cache_is_not_pickled(self):
        """Check lazy memory maps and worker-safe pickling."""
        _write_partition(self.root, [("shard.bin", list(range(10)))])
        dataset = LMDataset(self.root, "train", context_length=3)

        self.assertEqual(dataset._mmap_cache, {})
        dataset[0]
        self.assertIsInstance(dataset._mmap_cache[0], np.memmap)

        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(restored._mmap_cache, {})
        self.assertEqual(restored[0][0].tolist(), [0, 1, 2])

    def test_invalid_lengths_are_rejected(self):
        """Reject invalid context lengths and strides."""
        _write_partition(self.root, [("shard.bin", list(range(10)))])
        invalid_values = [
            (0, None),
            (-1, None),
            (True, None),
            (4, 0),
            (4, -2),
            (4, True),
            (1.5, None),
        ]
        for context_length, stride in invalid_values:
            with self.subTest(context_length=context_length, stride=stride):
                with self.assertRaises(ValueError):
                    LMDataset(
                        self.root,
                        "train",
                        context_length=context_length,
                        stride=stride,
                    )

    def test_out_of_range_indices_are_rejected(self):
        """Reject negative and past-the-end dataset indices."""
        _write_partition(self.root, [("shard.bin", list(range(10)))])
        dataset = LMDataset(self.root, "train", context_length=3)

        with self.assertRaises(IndexError):
            dataset[-1]
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_shard_needs_input_and_next_token_target(self):
        """A training window needs context_length plus one token."""
        short_root = self.root / "short"
        _write_partition(short_root, [("short.bin", [1, 2, 3])])
        self.assertEqual(len(LMDataset(short_root, "train", context_length=3)), 0)




    def test_missing_manifest_is_rejected(self):
        """Reject a partition without a manifest."""
        with self.assertRaises(FileNotFoundError):
            LMDataset(self.root, "train", context_length=4)


class TestTokenShardWriter(unittest.TestCase):
    def setUp(self):
        """Create a temporary writer and tokenizer setup."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.tokenizer = _build_tokenizer(self.root / "tokenizer.json")

    def tearDown(self):
        """Remove temporary shard files."""
        self.temp_dir.cleanup()

    def test_round_trip(self):
        """Check serialized token IDs and corpus counts."""
        documents = ["Hello", " café", "\nlast"]
        output_dir = self.root / "train"
        manifest = write_token_shards(
            documents,
            output_dir,
            self.tokenizer,
            shard_size=5,
        )

        expected = []
        for document in documents:
            expected.extend(self.tokenizer.encode(document))
            expected.append(self.tokenizer.eos_id)
        actual = []
        for shard in manifest["shards"]:
            actual.extend(
                np.fromfile(output_dir / shard["file"], dtype=SHARD_DTYPE).tolist()
            )

        self.assertEqual(actual, expected)
        self.assertEqual(manifest["num_documents"], len(documents))
        self.assertEqual(manifest["num_chars"], sum(map(len, documents)))
        self.assertEqual(manifest["num_tokens"], len(expected))
        self.assertTrue(
            all(shard["num_tokens"] <= 5 for shard in manifest["shards"])
        )

    def test_empty_input(self):
        """Write a valid empty manifest without shards."""
        manifest = write_token_shards(
            [],
            self.root / "empty",
            self.tokenizer,
            shard_size=5,
        )

        self.assertEqual(manifest["shards"], [])
        self.assertEqual(manifest["num_tokens"], 0)
        self.assertTrue((self.root / "empty" / "manifest.json").is_file())

    def test_nonempty_output_is_rejected(self):
        """Do not overwrite an existing nonempty directory."""
        output_dir = self.root / "train"
        output_dir.mkdir()
        (output_dir / "keep.txt").write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "not empty"):
            write_token_shards(
                ["hello"],
                output_dir,
                self.tokenizer,
                shard_size=5,
            )

    def test_invalid_shard_size_is_rejected(self):
        """Reject nonpositive and Boolean shard sizes."""
        for shard_size in (0, -1, True):
            with self.subTest(shard_size=shard_size):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    write_token_shards(
                        ["hello"],
                        self.root / f"invalid-{shard_size}",
                        self.tokenizer,
                        shard_size=shard_size,
                    )

    def test_failed_write_leaves_no_partial_partition(self):
        """Keep failed writes atomic and allow a clean retry."""
        output_dir = self.root / "train"

        def failing_documents():
            """Yield once and then simulate a source failure."""
            yield "first document"
            raise RuntimeError("source failed")

        with self.assertRaisesRegex(RuntimeError, "source failed"):
            write_token_shards(
                failing_documents(),
                output_dir,
                self.tokenizer,
                shard_size=3,
            )

        self.assertFalse(output_dir.exists())
        manifest = write_token_shards(
            ["retry succeeds"],
            output_dir,
            self.tokenizer,
            shard_size=3,
        )
        self.assertTrue((output_dir / "manifest.json").is_file())
        self.assertGreater(manifest["num_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
