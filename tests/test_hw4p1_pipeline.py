"""Small end-to-end smoke test for the redesigned HW4P1 reference path."""

from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from hw4lib.data import GPT2Tokenizer, LMDataset, write_token_shards
from hw4lib.model import DecoderOnlyTransformer


def test_one_training_step_and_cached_score():
    """Run one end-to-end fixed-shard training and cache check."""
    tokenizer = GPT2Tokenizer()
    documents = [
        "Transformers predict the next token. " * 5,
        "Grouped-query attention stores fewer keys and values. " * 5,
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        root = Path(temp_dir)
        write_token_shards(
            documents,
            root / "train",
            tokenizer,
            shard_size=256,
        )
        dataset = LMDataset(root, "train", context_length=8)
        inputs, targets = next(iter(DataLoader(dataset, batch_size=2)))

        torch.manual_seed(0)
        model = DecoderOnlyTransformer(
            num_layers=2,
            d_model=16,
            num_heads=4,
            num_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            max_len=16,
            num_classes=tokenizer.vocab_size,
            weight_tying=True,
        )

        logits, attention = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert model.target_embedding.weight.grad is not None
        assert set(attention) == {"layer1_dec_self", "layer2_dec_self"}

        model.eval()
        with torch.inference_mode():
            full_logits, _ = model(inputs)
            prompt_logits, cache = model.score_cached(inputs[:, :4])
            next_logits, cache = model.score_cached(inputs[:, 4:5], cache)

        assert prompt_logits.shape == (2, tokenizer.vocab_size)
        assert torch.allclose(
            prompt_logits,
            full_logits[:, 3],
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.allclose(
            next_logits,
            full_logits[:, 4],
            atol=1e-5,
            rtol=1e-5,
        )
        assert cache[0][0].shape == (2, 2, 5, 4)

        if hasattr(dataset, "close"):
            dataset.close()


if __name__ == "__main__":
    test_one_training_step_and_cached_score()
    print("HW4P1 pipeline smoke test passed.")