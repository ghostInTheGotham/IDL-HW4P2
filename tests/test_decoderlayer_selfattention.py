import torch


def test_decoderlayer_selfattention(decoder_layer) -> None:
    '''
    Test the self-attention decoder layer.

    Args:
        decoder_layer: Decoder-layer class under test.

    Returns:
        None.
    '''
    # Structural Test
    test_initialization(decoder_layer)

    # Integration Test
    test_forward_shapes(decoder_layer)
    test_sublayer_integration(decoder_layer)


def test_initialization(decoder_layer) -> None:
    '''
    Test decoder-layer initialization.

    Args:
        decoder_layer: Decoder-layer class under test.

    Returns:
        None.
    '''
    print("Testing initialization ...")

    d_model = 32
    num_heads = 4
    d_ff = 64
    dropout = 0.1

    layer = decoder_layer(d_model, num_heads, d_ff, dropout)

    # Check if sublayers exist
    assert hasattr(layer, 'self_attn'), "Self-attention layer is missing"
    assert hasattr(layer, 'ffn'), "Feed-forward layer is missing"

    # Check if sublayers are not None
    assert layer.self_attn is not None, "Self-attention layer is None"
    assert layer.ffn is not None, "Feed-forward layer is None"

    print("Test Passed: All sublayers exist and are initialized correctly")


def test_forward_shapes(decoder_layer) -> None:
    '''
    Test decoder-layer forward shapes.

    Args:
        decoder_layer: Decoder-layer class under test.

    Returns:
        None.
    '''
    print("Testing forward shapes ...")

    batch_size = 2
    seq_len = 10
    d_model = 32
    num_heads = 4
    d_ff = 64
    dropout = 0.0  # For deterministic testing

    layer = decoder_layer(d_model, num_heads, d_ff, dropout)

    # Create test inputs
    x = torch.randn(batch_size, seq_len, d_model)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    # Forward pass
    output, attention_weights = layer(x, key_padding_mask, attn_mask)

    # Check shapes
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert attention_weights.shape == (batch_size, seq_len, seq_len), \
        f"Expected attention shape {(batch_size, seq_len, seq_len)}, got {attention_weights.shape}"

    print("Test Passed: Forward shapes are as expected")


def test_sublayer_integration(decoder_layer) -> None:
    '''
    Test decoder sublayer interaction.

    Args:
        decoder_layer: Decoder-layer class under test.

    Returns:
        None.
    '''
    print("Testing sublayer interaction ...")

    batch_size = 2
    seq_len = 10
    d_model = 32
    num_heads = 4
    d_ff = 64
    dropout = 0.0  # For deterministic testing

    layer = decoder_layer(d_model, num_heads, d_ff, dropout)

    # Create test inputs
    x = torch.randn(batch_size, seq_len, d_model)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    # Track intermediate outputs
    self_attn_output = None
    ffn_output = None

    def hook_self_attn(module, input_tensor, output_tensor):
        """Capture the self-attention output.

        Args:
            module: Hooked module.
            input_tensor: Module inputs.
            output_tensor: Module output.

        Returns:
            Unchanged module output.
        """
        nonlocal self_attn_output
        self_attn_output = output_tensor[0]  # First element because sublayer1 returns tuple
        result = output_tensor
        return result

    def hook_ffn(module, input_tensor, output_tensor):
        """Capture the feed-forward output.

        Args:
            module: Hooked module.
            input_tensor: Module inputs.
            output_tensor: Module output.

        Returns:
            Unchanged module output.
        """
        nonlocal ffn_output
        ffn_output = output_tensor
        result = output_tensor
        return result

    # Register hooks
    handle_self_attn = layer.self_attn.register_forward_hook(hook_self_attn)
    handle_ffn = layer.ffn.register_forward_hook(hook_ffn)

    # Forward pass
    output, _ = layer(x, key_padding_mask, attn_mask)

    # Remove hooks
    handle_self_attn.remove()
    handle_ffn.remove()

    # Verify data flow
    assert self_attn_output is not None, "Self-attention was not executed"
    assert ffn_output is not None, "Feed-forward was not executed"
    assert torch.equal(output, ffn_output), "Final output should be from feed-forward layer"
    assert self_attn_output.shape == x.shape, "Self-attention output shape mismatch"
    assert ffn_output.shape == x.shape, "Feed-forward output shape mismatch"

    print("Test Passed: Sublayers interact correctly")


def test_rotary_cache_path(decoder_layer) -> None:
    '''
    Test GQA causal attention and cached/full equivalence.

    Args:
        decoder_layer: Decoder-layer class under test.

    Returns:
        None.
    '''
    print("Testing rotary attention and KV caching ...")

    layer = decoder_layer(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        d_ff=32,
        dropout=0.0,
        max_len=16,
        use_rope=True,
    ).eval()

    torch.manual_seed(0)
    x = torch.randn(2, 6, 16)
    full_output, full_attention = layer(x)
    assert full_output.shape == x.shape
    assert full_attention.shape == (2, 6, 6)
    assert torch.count_nonzero(torch.triu(full_attention, diagonal=1)) == 0

    cache = None
    step_outputs = []
    for position in range(x.size(1)):
        output, attention, cache = layer.forward_cached(
            x[:, position : position + 1],
            past_kv=cache,
        )
        step_outputs.append(output)
        assert attention.shape == (2, 1, position + 1)
        assert cache[0].shape == (2, 2, position + 1, 4)

    cached_output = torch.cat(step_outputs, dim=1)
    assert torch.allclose(full_output, cached_output, atol=1e-5, rtol=1e-5)
    print("Test Passed: Cached and full rotary outputs match")


def main() -> None:
    '''
    Run decoder-layer tests with the repository test framework.

    Returns:
        None.
    '''
    from hw4lib.model import SelfAttentionDecoderLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'SelfAttentionDecoderLayer': [
                {
                    'func': lambda: test_decoderlayer_selfattention(SelfAttentionDecoderLayer),
                    'description': 'Test the self-attention decoder layer'
                },
                {
                    'func': lambda: test_rotary_cache_path(SelfAttentionDecoderLayer),
                    'description': 'Test rotary attention and KV caching'
                }
            ]
        }
    )
    framework.run_tests()
    framework.summarize_results()
    framework.raise_if_failed()


if __name__ == '__main__':
    main()
