import inspect

import torch

from hw4lib.decoding.sequence_generator import SequenceGenerator
from tests.decoding_test_utils import (
    HistoryScorer,
    TinyTokenizer,
    make_generator,
    repeated_token_logits,
)


def test_greedy_api_has_no_temperature():
    parameters = inspect.signature(SequenceGenerator.generate_greedy).parameters
    assert "temperature" not in parameters


def test_cached_greedy_matches_full_prefix():
    prompts = torch.tensor([[0, 1], [0, 2]])

    full_scorer = HistoryScorer()
    cached_scorer = HistoryScorer()
    full_generator = make_generator(full_scorer, max_length=5, use_cache=False)
    cached_generator = make_generator(cached_scorer, max_length=5, use_cache=True)

    full_sequences, full_scores = full_generator.generate_greedy(prompts.clone())
    cached_sequences, cached_scores = cached_generator.generate_greedy(
        prompts.clone()
    )

    assert torch.equal(cached_sequences, full_sequences)
    assert torch.allclose(cached_scores, full_scores)
    assert cached_scorer.cached_input_lengths == [2, 1, 1]
    assert full_scorer.full_input_lengths == [2, 3, 4]
    assert cached_sequences[0].tolist() == [0, 1, 3, 6, 6]

    first_step = torch.log_softmax(HistoryScorer.logits(prompts), dim=-1)[0, 3]
    second_history = torch.tensor([[0, 1, 3]])
    second_step = torch.log_softmax(
        HistoryScorer.logits(second_history), dim=-1
    )[0, TinyTokenizer.eos_id]
    assert torch.allclose(cached_scores[0], first_step + second_step)


def test_greedy_applies_repeat_penalty():
    generator = SequenceGenerator(
        score_fn=repeated_token_logits,
        tokenizer=TinyTokenizer(),
        max_length=2,
        device="cpu",
    )
    prompt = torch.tensor([[1]])

    unpenalized, _ = generator.generate_greedy(
        prompt.clone(),
        repeat_penalty=1.0,
    )
    penalized, _ = generator.generate_greedy(
        prompt.clone(),
        repeat_penalty=2.0,
    )

    assert unpenalized.tolist() == [[1, 1]]
    assert penalized.tolist() == [[1, 2]]


def test_greedy_ignores_eos_inside_prompt():
    prompt = torch.tensor([[0, TinyTokenizer.eos_id, 2]])
    scorer = HistoryScorer()
    generator = make_generator(scorer, max_length=4, use_cache=True)

    sequence, score = generator.generate_greedy(prompt.clone())

    expected = torch.tensor([[0, TinyTokenizer.eos_id, 2, 4]])
    assert torch.equal(sequence, expected)
    assert score.isfinite().all()
    assert scorer.cached_input_lengths == [3]


def test_greedy_validation_and_postprocessing():
    scorer = HistoryScorer()
    generator = make_generator(scorer, max_length=4, use_cache=False)
    prompt = torch.tensor([[0, 1]])

    for repeat_penalty in (0.0, -1.0):
        try:
            generator.generate_greedy(prompt, repeat_penalty=repeat_penalty)
        except ValueError:
            pass
        else:
            raise AssertionError("greedy search must reject non-positive penalties")

    try:
        SequenceGenerator(
            score_fn=scorer.score_full,
            tokenizer=TinyTokenizer(),
            max_length=4.5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("max_length must be an integer")

    packed = torch.tensor(
        [[0, TinyTokenizer.eos_id, 2, 4, TinyTokenizer.eos_id, 1]]
    )
    processed = generator.post_process_sequence(
        packed,
        TinyTokenizer(),
        start_index=3,
    )
    assert processed[0].tolist() == [0, TinyTokenizer.eos_id, 2, 4, 6]


if __name__ == "__main__":
    test_greedy_api_has_no_temperature()
    test_cached_greedy_matches_full_prefix()
    test_greedy_applies_repeat_penalty()
    test_greedy_ignores_eos_inside_prompt()
    test_greedy_validation_and_postprocessing()
    print("All greedy decoding tests passed.")
