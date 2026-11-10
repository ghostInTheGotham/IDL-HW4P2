"""Shared checks for RoPE, causal MHA/GQA/MQA, and KV caching."""

from contextlib import contextmanager
import math
import re

import torch

from hw4lib.model import CausalSelfAttention, RotaryPositionalEncoding


@contextmanager
def assert_raises(error_type, match):
    """Check an exception type and message pattern.

    Args:
        error_type: Expected exception class.
        match: Regular-expression pattern for the message.

    Yields:
        Control to the checked code block.
    """
    try:
        yield
    except error_type as error:
        assert re.search(match, str(error)), (
            f"Expected error matching {match!r}, but received {str(error)!r}"
        )
    else:
        raise AssertionError(f"Expected {error_type.__name__} to be raised")


def test_rotary_positional_encoding_values_and_offset():
    """Check known rotary values, offsets, and norm preservation."""
    rope = RotaryPositionalEncoding(head_dim=4, max_len=8)
    vector = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    # Position zero has angle zero at every frequency.
    assert torch.equal(rope(vector, start_pos=0), vector)

    rotated = rope(vector, start_pos=1)
    expected = torch.tensor(
        [[[[
            math.cos(1.0) - 2.0 * math.sin(1.0),
            math.sin(1.0) + 2.0 * math.cos(1.0),
            3.0 * math.cos(0.01) - 4.0 * math.sin(0.01),
            3.0 * math.sin(0.01) + 4.0 * math.cos(0.01),
        ]]]]
    )
    assert torch.allclose(rotated, expected, atol=1e-6, rtol=1e-6)

    # A rotation preserves the norm of each neighboring pair.
    original_pair_norms = vector.reshape(1, 1, 1, 2, 2).norm(dim=-1)
    rotated_pair_norms = rotated.reshape(1, 1, 1, 2, 2).norm(dim=-1)
    assert torch.allclose(original_pair_norms, rotated_pair_norms)


def test_rotary_positional_encoding_validation():
    """Check rotary dimension and context-length validation."""
    with assert_raises(ValueError, match="even"):
        RotaryPositionalEncoding(head_dim=3, max_len=8)
    with assert_raises(ValueError, match="max_len"):
        RotaryPositionalEncoding(head_dim=4, max_len=0)

    rope = RotaryPositionalEncoding(head_dim=4, max_len=4)
    with assert_raises(ValueError, match="shape"):
        rope(torch.randn(2, 3, 4))
    with assert_raises(ValueError, match="exceed"):
        rope(torch.randn(1, 2, 2, 4), start_pos=3)


def test_mha_gqa_mqa_shapes_and_causal_probabilities():
    """Check projection shapes and causal probabilities for all KV counts."""
    torch.manual_seed(0)
    batch_size, sequence_length, d_model, num_heads = 2, 5, 16, 4
    x = torch.randn(batch_size, sequence_length, d_model)

    for num_kv_heads in (4, 2, 1):
        attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_len=16,
            dropout=0.0,
        ).eval()

        output, probabilities, cache = attention(x)
        head_dim = d_model // num_heads

        assert attention.q_proj.weight.shape == (d_model, d_model)
        assert attention.k_proj.weight.shape == (
            num_kv_heads * head_dim,
            d_model,
        )
        assert attention.v_proj.weight.shape == (
            num_kv_heads * head_dim,
            d_model,
        )
        assert output.shape == (batch_size, sequence_length, d_model)
        assert probabilities.shape == (
            batch_size,
            sequence_length,
            sequence_length,
        )
        assert cache is None

        assert torch.isfinite(output).all()
        assert torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones(batch_size, sequence_length),
        )
        assert torch.count_nonzero(torch.triu(probabilities, diagonal=1)) == 0


def test_grouped_query_head_repetition():
    """Check that compact KV heads repeat into the correct query groups."""
    attention = CausalSelfAttention(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        max_len=8,
    )
    kv = torch.tensor([[[[1.0]], [[2.0]]]])
    repeated = attention._repeat_kv(kv)
    expected = torch.tensor([[[[1.0]], [[1.0]], [[2.0]], [[2.0]]]])
    assert torch.equal(repeated, expected)


def test_padding_and_attention_masks():
    """Check the preserved mask parameters on rotary attention."""
    torch.manual_seed(5)
    attention = CausalSelfAttention(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        max_len=8,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 4, 16)
    key_padding_mask = torch.zeros(2, 4, dtype=torch.bool)
    key_padding_mask[0, 3] = True
    attn_mask = torch.zeros(4, 4, dtype=torch.bool)
    attn_mask[3, 0] = True

    _, probabilities, _ = attention(
        x,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    assert probabilities[0, 3, 3] == 0
    assert probabilities[:, 3, 0].count_nonzero() == 0


def test_fully_masked_rows_have_zero_attention():
    """A query with no valid keys must not leak a uniform distribution."""
    torch.manual_seed(6)
    attention = CausalSelfAttention(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        max_len=8,
        dropout=0.0,
    ).eval()
    x = torch.randn(1, 2, 16)
    key_padding_mask = torch.ones(1, 2, dtype=torch.bool)

    output, probabilities, _ = attention(
        x,
        key_padding_mask=key_padding_mask,
    )

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(probabilities) == 0


def test_causal_prefix_does_not_depend_on_future_tokens():
    """Check that changing future tokens leaves earlier outputs unchanged."""
    torch.manual_seed(1)
    attention = CausalSelfAttention(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        max_len=8,
        dropout=0.0,
    ).eval()

    first = torch.randn(2, 5, 16)
    second = first.clone()
    second[:, 3:] = torch.randn_like(second[:, 3:])

    first_output, _, _ = attention(first)
    second_output, _, _ = attention(second)
    assert torch.allclose(
        first_output[:, :3],
        second_output[:, :3],
        atol=1e-6,
        rtol=1e-6,
    )


def test_cached_outputs_match_full_sequence_for_all_attention_types():
    """Compare token-at-a-time and full outputs for MHA, GQA, and MQA."""
    torch.manual_seed(2)
    x = torch.randn(2, 6, 16)

    for num_kv_heads in (4, 2, 1):
        attention = CausalSelfAttention(
            d_model=16,
            num_heads=4,
            num_kv_heads=num_kv_heads,
            max_len=8,
            dropout=0.0,
        ).eval()

        full_output, _, full_cache = attention(x, use_cache=True)

        cache = None
        step_outputs = []
        first_cache_copy = None
        for position in range(x.size(1)):
            step_output, probabilities, cache = attention(
                x[:, position : position + 1],
                past_kv=cache,
                use_cache=True,
            )
            step_outputs.append(step_output)
            assert probabilities.shape == (2, 1, position + 1)
            assert cache is not None
            assert cache[0].shape == (
                2,
                num_kv_heads,
                position + 1,
                4,
            )

            if position == 0:
                first_cache_copy = (cache[0].clone(), cache[1].clone())
            elif position == 1:
                assert torch.equal(cache[0][:, :, :1], first_cache_copy[0])
                assert torch.equal(cache[1][:, :, :1], first_cache_copy[1])

        incremental_output = torch.cat(step_outputs, dim=1)
        assert torch.allclose(full_output, incremental_output, atol=1e-5, rtol=1e-5)
        assert torch.allclose(full_cache[0], cache[0], atol=1e-6, rtol=1e-6)
        assert torch.allclose(full_cache[1], cache[1], atol=1e-6, rtol=1e-6)


def test_cached_chunk_uses_offset_causal_mask():
    """Check causal positions when a multi-token chunk extends a cache."""
    torch.manual_seed(3)
    attention = CausalSelfAttention(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        max_len=8,
        dropout=0.0,
    ).eval()

    prompt = torch.randn(1, 2, 16)
    chunk = torch.randn(1, 2, 16)
    _, _, cache = attention(prompt, use_cache=True)
    _, probabilities, cache = attention(chunk, past_kv=cache, use_cache=True)

    assert probabilities.shape == (1, 2, 4)
    assert probabilities[0, 0, 3] == 0
    assert probabilities[0, 0, :3].sum() > 0
    assert torch.allclose(probabilities[0, 1].sum(), torch.tensor(1.0))
    assert cache[0].shape == (1, 2, 4, 4)


def test_cache_stores_compact_kv_heads():
    """The cache must keep num_kv_heads, never the repeated num_heads."""
    torch.manual_seed(7)
    d_model, num_heads = 16, 4
    x = torch.randn(2, 5, d_model)

    for num_kv_heads in (4, 2, 1):
        attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_len=16,
            dropout=0.0,
        ).eval()

        _, _, cache = attention(x, use_cache=True)
        assert cache is not None

        for tensor in cache:
            assert tensor.size(1) == num_kv_heads, (
                f"cache holds {tensor.size(1)} heads, expected {num_kv_heads}"
            )
            # A cache written after _repeat_kv would come back num_heads wide.
            if num_kv_heads != num_heads:
                assert tensor.size(1) != num_heads, (
                    "the cache must be built before the key/value heads are repeated"
                )

        # Extending the cache must not widen it either.
        _, _, extended = attention(x[:, :1], past_kv=cache, use_cache=True)
        for tensor in extended:
            assert tensor.size(1) == num_kv_heads, (
                f"extended cache holds {tensor.size(1)} heads, expected {num_kv_heads}"
            )


def test_cache_storage_scales_linearly_with_kv_heads():
    """Measure the exact cache footprint for MHA, GQA, and MQA."""
    torch.manual_seed(8)
    batch_size, sequence_length = 2, 5
    d_model, num_heads = 16, 4
    head_dim = d_model // num_heads
    x = torch.randn(batch_size, sequence_length, d_model)
    cache_bytes = {}

    for num_kv_heads in (4, 2, 1):
        attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_len=16,
            dropout=0.0,
        ).eval()
        with torch.inference_mode():
            _, _, cache = attention(x, use_cache=True)
        assert cache is not None

        measured_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in cache
        )
        expected_bytes = (
            2
            * batch_size
            * num_kv_heads
            * sequence_length
            * head_dim
            * x.element_size()
        )
        assert measured_bytes == expected_bytes
        cache_bytes[num_kv_heads] = measured_bytes

    # Cache storage is proportional to compact KV heads, not query heads.
    assert cache_bytes[2] * 2 == cache_bytes[4]
    assert cache_bytes[1] * 4 == cache_bytes[4]


def test_cache_does_not_retain_autograd_history():
    """A progressive inference cache must not retain earlier graphs."""
    torch.manual_seed(9)
    attention = CausalSelfAttention(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        max_len=8,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 3, 16, requires_grad=True)

    output, _, cache = attention(x, use_cache=True)
    assert cache is not None
    for tensor in cache:
        assert not tensor.requires_grad
        assert tensor.grad_fn is None

    # Returning a detached cache must not detach the current output path.
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_attention_validation_and_cache_errors():
    """Check invalid dimensions, head counts, lengths, and caches."""
    with assert_raises(ValueError, match="divisible by num_heads"):
        CausalSelfAttention(d_model=10, num_heads=4, max_len=8)
    with assert_raises(ValueError, match="num_kv_heads"):
        CausalSelfAttention(
            d_model=16,
            num_heads=4,
            num_kv_heads=3,
            max_len=8,
        )
    with assert_raises(ValueError, match="even"):
        CausalSelfAttention(d_model=12, num_heads=4, max_len=8)

    attention = CausalSelfAttention(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        max_len=4,
    )
    x = torch.randn(1, 2, 16)
    _, _, cache = attention(x, use_cache=True)

    with assert_raises(ValueError, match="use_cache"):
        attention(x[:, :1], past_kv=cache, use_cache=False)
    with assert_raises(ValueError, match="exceeds max_len"):
        attention(torch.randn(1, 3, 16), past_kv=cache, use_cache=True)

    wrong_heads = (
        torch.randn(1, 1, 1, 4),
        torch.randn(1, 1, 1, 4),
    )
    with assert_raises(ValueError, match="KV heads"):
        attention(x[:, :1], past_kv=wrong_heads, use_cache=True)


def test_attention_backward_reaches_all_projections():
    """Check finite input and projection gradients."""
    torch.manual_seed(4)
    attention = CausalSelfAttention(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        max_len=8,
        dropout=0.0,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)

    output, _, cache = attention(x)
    assert cache is None
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for projection in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        attention.out_proj,
    ):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()


if __name__ == "__main__":
    test_mha_gqa_mqa_shapes_and_causal_probabilities()
    test_grouped_query_head_repetition()
    test_padding_and_attention_masks()
    test_fully_masked_rows_have_zero_attention()
    test_causal_prefix_does_not_depend_on_future_tokens()
    test_attention_backward_reaches_all_projections()
    print("All causal self-attention tests passed.")
