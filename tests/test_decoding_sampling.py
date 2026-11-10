import inspect

import torch

from hw4lib.decoding.sequence_generator import SequenceGenerator
from tests.decoding_test_utils import (
    HistoryScorer,
    TinyTokenizer,
    make_generator,
    repeated_token_logits,
    sampling_logits,
)


def test_sampling_api_has_temperature_and_seed():
    parameters = inspect.signature(SequenceGenerator.generate_sample).parameters
    assert "temperature" in parameters
    assert "seed" in parameters


def test_filter_logits_temperature_top_k_and_top_p():
    generator = SequenceGenerator(
        score_fn=sampling_logits,
        tokenizer=TinyTokenizer(),
        max_length=2,
        device="cpu",
    )

    logits = torch.tensor([[2.0, 1.0, 0.0]])
    filtered = generator._filter_logits(
        logits.clone(),
        temperature=2.0,
        top_p=0.6,
    )
    assert torch.allclose(filtered[0, :2], torch.tensor([1.0, 0.5]))
    assert torch.isneginf(filtered[0, 2])

    top_k_filtered = generator._filter_logits(
        torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
        top_k=2,
    )
    assert torch.isfinite(top_k_filtered).sum().item() == 2

    oversized_top_k = generator._filter_logits(
        logits.clone(),
        top_k=10,
    )
    assert torch.equal(oversized_top_k, logits)

    crossing = generator._filter_logits(
        torch.tensor([[3.0, 2.0, 1.0, 0.0]]),
        top_p=0.8,
    )
    assert torch.isfinite(crossing[0, :2]).all()
    assert torch.isneginf(crossing[0, 2:]).all()


def test_seeded_sampling_is_reproducible_and_uses_local_rng():
    generator = SequenceGenerator(
        score_fn=sampling_logits,
        tokenizer=TinyTokenizer(),
        max_length=8,
        device="cpu",
    )
    prompt = torch.tensor([[0]])

    torch.manual_seed(2026)
    global_state = torch.random.get_rng_state().clone()
    first_sequence, first_score = generator.generate_sample(
        prompt.clone(),
        seed=11785,
    )
    assert torch.equal(torch.random.get_rng_state(), global_state)

    torch.manual_seed(99)
    second_sequence, second_score = generator.generate_sample(
        prompt.clone(),
        seed=11785,
    )
    assert torch.equal(first_sequence, second_sequence)
    assert torch.allclose(first_score, second_score)

    different_sequence, _ = generator.generate_sample(
        prompt.clone(),
        seed=11786,
    )
    assert not torch.equal(first_sequence, different_sequence)


def test_cached_sampling_matches_full_prefix():
    prompts = torch.tensor([[0, 1], [0, 2]])
    full_scorer = HistoryScorer()
    cached_scorer = HistoryScorer()
    full_generator = make_generator(full_scorer, max_length=5, use_cache=False)
    cached_generator = make_generator(cached_scorer, max_length=5, use_cache=True)

    full_sequences, full_scores = full_generator.generate_sample(
        prompts.clone(),
        temperature=0.7,
        top_k=3,
        top_p=0.95,
        seed=11785,
    )
    cached_sequences, cached_scores = cached_generator.generate_sample(
        prompts.clone(),
        temperature=0.7,
        top_k=3,
        top_p=0.95,
        seed=11785,
    )

    assert torch.equal(cached_sequences, full_sequences)
    assert torch.allclose(cached_scores, full_scores)
    assert cached_scorer.cached_input_lengths == [2, 1, 1]
    assert full_scorer.full_input_lengths == [2, 3, 4]


def test_sampling_applies_repeat_penalty():
    generator = SequenceGenerator(
        score_fn=repeated_token_logits,
        tokenizer=TinyTokenizer(),
        max_length=2,
        device="cpu",
    )
    prompt = torch.tensor([[1]])

    unpenalized, _ = generator.generate_sample(
        prompt.clone(),
        top_k=1,
        repeat_penalty=1.0,
        seed=11785,
    )
    penalized, _ = generator.generate_sample(
        prompt.clone(),
        top_k=1,
        repeat_penalty=2.0,
        seed=11785,
    )

    assert unpenalized.tolist() == [[1, 1]]
    assert penalized.tolist() == [[1, 2]]


def test_sampling_ignores_eos_inside_prompt():
    prompt = torch.tensor([[0, TinyTokenizer.eos_id, 2]])
    scorer = HistoryScorer()
    generator = make_generator(scorer, max_length=4, use_cache=True)

    sequence, score = generator.generate_sample(
        prompt.clone(),
        top_k=1,
        seed=11785,
    )

    expected = torch.tensor([[0, TinyTokenizer.eos_id, 2, 4]])
    assert torch.equal(sequence, expected)
    assert score.isfinite().all()
    assert scorer.cached_input_lengths == [3]


def test_sampling_argument_validation():
    scorer = HistoryScorer()
    generator = make_generator(scorer, max_length=4, use_cache=False)
    prompt = torch.tensor([[0, 1]])

    invalid_arguments = [
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"top_k": -1},
        {"top_k": 1.5},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"repeat_penalty": 0.0},
        {"seed": True},
        {"seed": "11785"},
    ]
    for arguments in invalid_arguments:
        try:
            generator.generate_sample(prompt, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sampling must reject {arguments}")


if __name__ == "__main__":
    test_sampling_api_has_temperature_and_seed()
    test_filter_logits_temperature_top_k_and_top_p()
    test_seeded_sampling_is_reproducible_and_uses_local_rng()
    test_cached_sampling_matches_full_prefix()
    test_sampling_applies_repeat_penalty()
    test_sampling_ignores_eos_inside_prompt()
    test_sampling_argument_validation()
    print("All sampling decoding tests passed.")
