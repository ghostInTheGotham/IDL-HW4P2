from collections.abc import Callable
from contextlib import contextmanager
import warnings

import numpy as np
import json
from pathlib import Path

from mytorch.nn import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)


@contextmanager
def ignore_macos_matmul_warnings():
    """Ignore false-positive Accelerate matmul warnings in this test process."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        yield


def run_with_macos_matmul_warnings_ignored(
    test_function: Callable[[], None],
) -> None:
    """Run one legacy NumPy test with the narrow macOS warning filter.

    Args:
        test_function: Zero-argument test function to run.

    Returns:
        None.
    """
    with ignore_macos_matmul_warnings():
        test_function()


def _make_weights(
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Create deterministic projection parameters for one attention module.

    Args:
        embed_dim: Size of the input and output embeddings.
        num_heads: Number of query heads.
        num_kv_heads: Number of key and value heads.
        seed: Seed used by the random number generator.

    Returns:
        Query, key, value, and output weights and biases in ``init_weights``
        argument order.
    """
    generator = np.random.default_rng(seed)
    head_dim = embed_dim // num_heads
    kv_dim = num_kv_heads * head_dim

    Wq = generator.normal(scale=0.2, size=(embed_dim, embed_dim)).astype(np.float32)
    bq = generator.normal(scale=0.2, size=(embed_dim,)).astype(np.float32)
    Wk = generator.normal(scale=0.2, size=(kv_dim, embed_dim)).astype(np.float32)
    bk = generator.normal(scale=0.2, size=(kv_dim,)).astype(np.float32)
    Wv = generator.normal(scale=0.2, size=(kv_dim, embed_dim)).astype(np.float32)
    bv = generator.normal(scale=0.2, size=(kv_dim,)).astype(np.float32)
    Wo = generator.normal(scale=0.2, size=(embed_dim, embed_dim)).astype(np.float32)
    bo = generator.normal(scale=0.2, size=(embed_dim,)).astype(np.float32)

    weights = (Wq, bq, Wk, bk, Wv, bv, Wo, bo)
    return weights


def test_grouped_query_attention_forward_and_backward() -> None:
    """Compare MHA, GQA, and MQA outputs and gradients with posted numeric cases."""
    fixture_path = Path(__file__).with_name("gqa_reference.json")
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        query, key, value, d_output = (
            np.asarray(case[name], dtype=np.float32)
            for name in ("query", "key", "value", "d_output")
        )
        weights = tuple(np.asarray(weight, dtype=np.float32)
                        for weight in case["weights"])
        key_padding_mask = np.asarray(case["key_padding_mask"], dtype=bool)
        attn_mask = np.asarray(case["attn_mask"], dtype=bool)
        attention = GroupedQueryAttention(
            embed_dim=query.shape[-1],
            num_heads=case["num_heads"],
            num_kv_heads=case["num_kv_heads"],
        )
        attention.init_weights(*weights)
        with ignore_macos_matmul_warnings():
            output = attention.forward(
                query, key, value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            d_query, d_key, d_value = attention.backward(d_output)
            input_gradients = (d_query, d_key, d_value)
        assert np.isfinite(output).all()
        np.testing.assert_allclose(
            output, np.asarray(case["output"]), rtol=1e-5, atol=1e-5,
        )
        for actual, expected in zip(input_gradients, case["input_gradients"]):
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(
                actual, np.asarray(expected), rtol=1e-5, atol=1e-5,
            )
        actual_parameter_gradients = (
            attention.q_proj.dLdW, attention.q_proj.dLdb,
            attention.k_proj.dLdW, attention.k_proj.dLdb,
            attention.v_proj.dLdW, attention.v_proj.dLdb,
            attention.out_proj.dLdW, attention.out_proj.dLdb,
        )
        for actual, expected in zip(
            actual_parameter_gradients, case["parameter_gradients"],
        ):
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(
                actual, np.asarray(expected), rtol=1e-5, atol=1e-5,
            )


def test_attention_specializations_and_optional_masks() -> None:
    """Check the MHA/MQA wrappers and each optional-mask path.

    Args:
        None.

    Returns:
        None.
    """
    generator = np.random.default_rng(7)
    embed_dim = 8
    num_heads = 4
    inputs = generator.normal(size=(2, 3, embed_dim)).astype(np.float32)
    key_padding_mask = np.array(
        [[False, False, True], [False, True, False]]
    )
    attn_mask = np.triu(np.ones((3, 3), dtype=bool), k=1)

    specializations = (
        (MultiHeadAttention(embed_dim, num_heads), num_heads),
        (MultiQueryAttention(embed_dim, num_heads), 1),
    )
    for specialized_attention, num_kv_heads in specializations:
        generic_attention = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        weights = _make_weights(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seed=num_kv_heads,
        )
        specialized_attention.init_weights(*weights)
        generic_attention.init_weights(*weights)

        with ignore_macos_matmul_warnings():
            specialized_output = specialized_attention.forward(
                inputs,
                inputs,
                inputs,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            generic_output = generic_attention.forward(
                inputs,
                inputs,
                inputs,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        np.testing.assert_allclose(specialized_output, generic_output)

        with ignore_macos_matmul_warnings():
            no_mask_output = specialized_attention.forward(inputs, inputs, inputs)
            padding_only_output = specialized_attention.forward(
                inputs,
                inputs,
                inputs,
                key_padding_mask=key_padding_mask,
            )
            attention_only_output = specialized_attention.forward(
                inputs,
                inputs,
                inputs,
                attn_mask=attn_mask,
            )
        assert np.isfinite(no_mask_output).all()
        assert np.isfinite(padding_only_output).all()
        assert np.isfinite(attention_only_output).all()


def test_grouped_query_attention_validation_and_shapes() -> None:
    """Check head-count validation and compact projection dimensions.

    Args:
        None.

    Returns:
        None.
    """
    invalid_arguments = (
        (0, 4, 2),
        (8, 0, 1),
        (8, 4, 0),
        (10, 4, 2),
        (8, 4, 3),
    )
    for arguments in invalid_arguments:
        raised_error = False
        try:
            GroupedQueryAttention(*arguments)
        except ValueError:
            raised_error = True
        assert raised_error

    attention = GroupedQueryAttention(
        embed_dim=16,
        num_heads=8,
        num_kv_heads=2,
    )
    assert attention.q_proj.W.shape == (16, 16)
    assert attention.k_proj.W.shape == (4, 16)
    assert attention.v_proj.W.shape == (4, 16)
    assert attention.out_proj.W.shape == (16, 16)

    compact_heads = np.arange(8).reshape(1, 2, 2, 2)
    repeated_heads = attention._repeat_kv(compact_heads)
    expected_repeated = np.repeat(compact_heads, 4, axis=1)
    np.testing.assert_array_equal(repeated_heads, expected_repeated)

    repeated_gradients = np.ones((1, 8, 2, 2))
    compact_gradients = attention._sum_kv_gradients(repeated_gradients)
    expected_gradients = np.full((1, 2, 2, 2), 4.0)
    np.testing.assert_array_equal(compact_gradients, expected_gradients)


def test_fully_masked_rows_have_zero_probabilities() -> None:
    """Check that MHA, GQA, and MQA do not attend through an empty row.

    Args:
        None.

    Returns:
        None.
    """
    generator = np.random.default_rng(19)
    inputs = generator.normal(size=(2, 3, 8)).astype(np.float32)
    key_padding_mask = np.ones((2, 3), dtype=bool)

    attention_modules = (
        MultiHeadAttention(embed_dim=8, num_heads=4),
        GroupedQueryAttention(embed_dim=8, num_heads=4, num_kv_heads=2),
        MultiQueryAttention(embed_dim=8, num_heads=4),
    )
    for index, attention in enumerate(attention_modules):
        weights = _make_weights(
            embed_dim=8,
            num_heads=4,
            num_kv_heads=attention.num_kv_heads
            if hasattr(attention, "num_kv_heads")
            else 4,
            seed=200 + index,
        )
        attention.init_weights(*weights)
        with ignore_macos_matmul_warnings():
            output = attention.forward(
                inputs,
                inputs,
                inputs,
                key_padding_mask=key_padding_mask,
            )
        probabilities = attention.attention.attention_scores
        assert np.isfinite(output).all()
        assert np.count_nonzero(probabilities) == 0


if __name__ == "__main__":
    test_grouped_query_attention_forward_and_backward()
    test_attention_specializations_and_optional_masks()
    test_grouped_query_attention_validation_and_shapes()
    test_fully_masked_rows_have_zero_probabilities()
    print("All grouped-query attention tests passed.")
