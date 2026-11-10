from contextlib import contextmanager
import math

import torch


def test_decoder_only_transformer(transformer) -> None:
    '''
    Test the decoder only transformer.

    Args:
        transformer: Decoder-only Transformer class under test.

    Returns:
        None.
    '''

    # Structural tests
    test_initialization(transformer)

    # Integration tests
    test_forward_pass(transformer)
    test_forward_propagation_order(transformer)


def test_initialization(transformer) -> None:
    '''
    Test the initialization of the decoder only transformer.

    Args:
        transformer: Decoder-only Transformer class under test.

    Returns:
        None.
    '''
    print("Testing initialization...")

    # Test parameters
    d_model = 16
    num_heads = 4
    d_ff = 32
    num_layers = 2
    dropout = 0.1
    max_len = 100
    num_classes = 10

    model = transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes
    )

    # Check if all required components exist
    expected_attributes = {
        "dec_layers", "target_embedding", "positional_encoding",
        "final_linear", "dropout", "norm", "max_len", "num_classes", "num_layers"
    }
    assert expected_attributes.issubset(dir(model)), "Required components are missing"

    # Check all attributes are not None
    for attr in expected_attributes:
        assert getattr(model, attr) is not None, f"{attr} is None"

    # Check number of decoder layers
    assert len(model.dec_layers) == num_layers, f"Expected {num_layers} decoder layers, got {len(model.dec_layers)}"

    # Check embedding dimensions
    assert model.target_embedding.embedding_dim == d_model, "Embedding dimension mismatch"
    assert model.target_embedding.num_embeddings == num_classes, "Vocabulary size mismatch"

    print("Test Passed: All components initialized correctly")


def test_forward_pass(transformer) -> None:
    '''
    Test the forward pass of the decoder only transformer.

    Args:
        transformer: Decoder-only Transformer class under test.

    Returns:
        None.
    '''
    print("Testing forward pass...")

    # Test parameters
    batch_size = 2
    seq_length = 8
    d_model = 16
    num_heads = 4
    d_ff = 32
    num_layers = 2
    dropout = 0.0  # For deterministic testing
    max_len = 100
    num_classes = 10

    model = transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes
    )

    # Create test inputs
    targets = torch.randint(0, num_classes, (batch_size, seq_length))
    target_lengths = torch.ones(batch_size, dtype=torch.int32) * seq_length

    # Forward pass
    output, attention_weights = model(targets, target_lengths)

    # Check output shape
    expected_shape = (batch_size, seq_length, num_classes)
    assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"

    # Check attention weights
    for i in range(num_layers):
        attn_key = f'layer{i+1}_dec_self'
        assert attn_key in attention_weights, f"Missing attention weights for {attn_key}"
        assert attention_weights[attn_key].shape == (batch_size, seq_length, seq_length), \
            f"Attention weights shape mismatch for {attn_key}"

    print("Test Passed: Forward pass works correctly")


def test_forward_propagation_order(transformer) -> None:
    '''
    Test that modules are called in the correct order during forward propagation.

    Args:
        transformer: Decoder-only Transformer class under test.

    Returns:
        None.
    '''
    print("Testing forward propagation order...")

    # Test parameters
    batch_size = 2
    seq_length = 8
    d_model = 16
    num_heads = 4
    d_ff = 32
    num_layers = 2

    model = transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=100,
        num_classes=10,
        use_rope=False,
    )

    # Create test inputs
    targets = torch.randint(0, 10, (batch_size, seq_length))
    target_lengths = torch.ones(batch_size, dtype=torch.int32) * seq_length

    # Track module execution order
    execution_order = []

    def make_hook(name):
        """Create a hook that records one module name.

        Args:
            name: Module name to record.

        Returns:
            Forward hook function.
        """
        def hook(module, input_tensor, output_tensor):
            """Record one module execution.

            Args:
                module: Hooked module.
                input_tensor: Module inputs.
                output_tensor: Module output.

            Returns:
                Unchanged module output.
            """
            execution_order.append(name)
            result = output_tensor
            return result
        created_hook = hook
        return created_hook

    # Register hooks for all relevant modules
    handles = [
        model.target_embedding.register_forward_hook(make_hook('target_embedding')),
        model.positional_encoding.register_forward_hook(make_hook('pos_encoding')),
        model.dropout.register_forward_hook(make_hook('dropout')),
    ]

    # Add hooks for decoder layers
    for i, layer in enumerate(model.dec_layers):
        handles.append(layer.register_forward_hook(make_hook(f'decoder_layer_{i}')))

    # Add hooks for final operations
    handles.extend([
        model.norm.register_forward_hook(make_hook('final_norm')),
        model.final_linear.register_forward_hook(make_hook('final_linear'))
    ])

    # Forward pass
    model(targets, target_lengths)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Expected order based on implementation
    expected_order = [
        'target_embedding',
        'pos_encoding',
        'dropout',
        'decoder_layer_0',
        'decoder_layer_1',
        'final_norm',
        'final_linear'
    ]

    # Verify execution order
    assert execution_order == expected_order, \
        f"Incorrect execution order.\nExpected: {expected_order}\nGot: {execution_order}"

    print("Test Passed: Forward propagation order is correct")


@contextmanager
def assert_raises(error_type):
    """Assert that a block raises the requested error type.

    Args:
        error_type: Expected exception class.

    Yields:
        Control to the checked code block.
    """
    try:
        yield
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__} to be raised")


def make_model(transformer, num_kv_heads=2, weight_tying=False):
    """Build a small decoder-only model for new HW4P1 checks.

    Args:
        transformer: Decoder-only Transformer class under test.
        num_kv_heads: Number of compact key/value heads.
        weight_tying: Whether to share embedding and output weights.

    Returns:
        Configured model.
    """
    model = transformer(
        num_layers=2,
        d_model=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        d_ff=32,
        dropout=0.0,
        max_len=16,
        num_classes=31,
        weight_tying=weight_tying,
    )
    return model


def test_attention_types_and_cache(transformer) -> None:
    """Check MHA/GQA/MQA, causal outputs, and cached equivalence.

    Args:
        transformer: Decoder-only Transformer class under test.

    Returns:
        None.
    """
    print("Testing MHA, GQA, MQA, and cached decoding...")

    torch.manual_seed(0)
    targets = torch.randint(0, 31, (2, 6))

    for num_kv_heads in (4, 2, 1):
        model = make_model(transformer, num_kv_heads=num_kv_heads).eval()
        full_logits, attention_weights = model(targets)

        assert full_logits.shape == (2, 6, 31)
        assert set(attention_weights) == {"layer1_dec_self", "layer2_dec_self"}
        for weights in attention_weights.values():
            assert weights.shape == (2, 6, 6)
            assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0

        cache = None
        cached_logits = []
        for position in range(targets.size(1)):
            logits, cache = model.score_cached(
                targets[:, position : position + 1],
                past_key_values=cache,
            )
            cached_logits.append(logits)
            assert len(cache) == model.num_layers
            for key, value in cache:
                assert key.shape == (2, num_kv_heads, position + 1, 4)
                assert value.shape == key.shape

        cached_logits = torch.stack(cached_logits, dim=1)
        assert torch.allclose(full_logits, cached_logits, atol=1e-5, rtol=1e-5)

    print("Test Passed: All attention types match cached decoding")


def test_weight_tying_and_validation(transformer) -> None:
    """Check weight tying, fixed lengths, and inference-only caching.

    Args:
        transformer: Decoder-only Transformer class under test.

    Returns:
        None.
    """
    model = make_model(transformer, weight_tying=True)
    assert model.target_embedding.weight.data_ptr() == model.final_linear.weight.data_ptr()

    targets = torch.randint(0, 31, (2, 4))
    target_lengths = torch.tensor([4, 3])
    with assert_raises(ValueError):
        model(targets, target_lengths=target_lengths)

    model.eval()
    with assert_raises(ValueError):
        model.forward_cached(targets, past_key_values=((torch.empty(0), torch.empty(0)),))

    model.train()
    with assert_raises(ValueError):
        model.score_cached(targets)


def test_gpt_initialization(transformer) -> None:
    """Check GPT initialization and depth-scaled residual outputs."""
    torch.manual_seed(11785)
    num_layers = 4
    model = transformer(
        num_layers=num_layers,
        d_model=128,
        num_heads=8,
        num_kv_heads=2,
        d_ff=256,
        dropout=0.0,
        max_len=16,
        num_classes=512,
        weight_tying=False,
    )

    def assert_std(tensor, expected):
        observed = tensor.detach().float().std(unbiased=False).item()
        tolerance = expected * 0.15
        assert abs(observed - expected) < tolerance, (
            f"Expected std near {expected:.6f}, got {observed:.6f}"
        )

    initialization_std = 0.02
    residual_std = initialization_std / math.sqrt(2 * num_layers)
    assert_std(model.target_embedding.weight, initialization_std)
    assert_std(model.final_linear.weight, initialization_std)

    for layer in model.dec_layers:
        assert_std(layer.self_attn.mha.q_proj.weight, initialization_std)
        assert_std(layer.self_attn.mha.out_proj.weight, residual_std)
        assert_std(layer.ffn.ffn[0].weight, initialization_std)
        assert_std(layer.ffn.ffn[3].weight, residual_std)

    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.equal(module.bias, torch.zeros_like(module.bias))

    tied_model = transformer(
        num_layers=1,
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        d_ff=32,
        dropout=0.0,
        max_len=16,
        num_classes=31,
        weight_tying=True,
    )
    assert (
        tied_model.target_embedding.weight.data_ptr()
        == tied_model.final_linear.weight.data_ptr()
    )


def test_model_level_cache_validation(transformer) -> None:
    """Reject invalid cache counts, shapes, dtypes, and lengths.

    Args:
        transformer: Decoder-only Transformer class under test.

    Returns:
        None.
    """
    model = make_model(transformer, num_kv_heads=2).eval()
    new_targets = torch.randint(0, 31, (2, 1))

    def cache_pair(shape, dtype=torch.float32):
        """Create a zero key/value cache pair.

        Args:
            shape: Key/value tensor shape.
            dtype: Key/value tensor dtype.

        Returns:
            Key and value tensors.
        """
        key = torch.zeros(shape, dtype=dtype)
        value = torch.zeros(shape, dtype=dtype)
        cache = (key, value)
        return cache

    invalid_caches = [
        # Decoder layers must agree on the past sequence length.
        (cache_pair((2, 2, 2, 4)), cache_pair((2, 2, 3, 4))),
        # Batch, KV-head, head-dimension, and dtype contracts are layer-local.
        tuple(cache_pair((1, 2, 2, 4)) for _ in range(2)),
        tuple(cache_pair((2, 1, 2, 4)) for _ in range(2)),
        tuple(cache_pair((2, 2, 2, 3)) for _ in range(2)),
        tuple(cache_pair((2, 2, 2, 4), torch.float64) for _ in range(2)),
        # The combined cache and new tokens must fit inside max_len.
        tuple(cache_pair((2, 2, model.max_len, 4)) for _ in range(2)),
    ]

    for invalid_cache in invalid_caches:
        with assert_raises(ValueError):
            model.forward_cached(new_targets, past_key_values=invalid_cache)


def test_additive_position_compatibility(transformer) -> None:
    """Check the preserved additive-position and mask path.

    Args:
        transformer: Decoder-only Transformer class under test.

    Returns:
        None.
    """
    with assert_raises(ValueError):
        transformer(
            num_layers=1,
            d_model=16,
            num_heads=4,
            num_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            max_len=16,
            num_classes=31,
            use_rope=False,
        )

    model = transformer(
        num_layers=1,
        d_model=16,
        num_heads=4,
        d_ff=32,
        dropout=0.0,
        max_len=16,
        num_classes=31,
        use_rope=False,
    ).eval()

    calls = []

    def record_position(module, inputs, output):
        """Record that additive positional encoding ran.

        Args:
            module: Hooked positional module.
            inputs: Module inputs.
            output: Module output.

        Returns:
            None.
        """
        calls.append("position")

    hook = model.positional_encoding.register_forward_hook(record_position)
    targets = torch.randint(0, 31, (2, 6))
    target_lengths = torch.tensor([6, 4])
    logits, attention_weights = model(
        padded_targets=targets,
        target_lengths=target_lengths,
    )
    hook.remove()

    assert calls == ["position"]
    assert logits.shape == (2, 6, 31)
    assert attention_weights["layer1_dec_self"].shape == (2, 6, 6)
    assert torch.count_nonzero(attention_weights["layer1_dec_self"][1, :, 4:]) == 0
    assert isinstance(model.dec_layers[0].self_attn.mha, torch.nn.MultiheadAttention)

    with assert_raises(ValueError):
        model.forward_cached(targets)


def main() -> None:
    '''
    Main function to run the tests.

    Returns:
        None.
    '''
    from hw4lib.model import DecoderOnlyTransformer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'DecoderOnlyTransformer': [
                {
                    'func': lambda: test_decoder_only_transformer(DecoderOnlyTransformer),
                    'description': 'Test the decoder only transformer'
                },
                {
                    'func': lambda: test_attention_types_and_cache(DecoderOnlyTransformer),
                    'description': 'Test MHA, GQA, MQA, and cached decoding'
                },
                {
                    'func': lambda: test_weight_tying_and_validation(DecoderOnlyTransformer),
                    'description': 'Test weight tying and validation'
                },
                {
                    'func': lambda: test_gpt_initialization(DecoderOnlyTransformer),
                    'description': 'Test GPT initialization and residual scaling'
                },
                {
                    'func': lambda: test_model_level_cache_validation(DecoderOnlyTransformer),
                    'description': 'Test model-level KV cache validation'
                },
                {
                    'func': lambda: test_additive_position_compatibility(DecoderOnlyTransformer),
                    'description': 'Test the additive positional compatibility path'
                }
            ]
        }
    )
    framework.run_tests()
    framework.summarize_results()
    framework.raise_if_failed()


if __name__ == '__main__':
    main()
