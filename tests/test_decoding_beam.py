import inspect

import torch

from hw4lib.decoding.sequence_generator import SequenceGenerator
from tests.decoding_test_utils import (
    BeamScorer,
    HistoryScorer,
    TinyTokenizer,
)


def test_beam_api_has_no_temperature():
    parameters = inspect.signature(SequenceGenerator.generate_beam).parameters
    assert "temperature" not in parameters


def test_beam_uses_full_prefix_scorer():
    scorer = HistoryScorer()

    def fail_cache_score(*_args):
        raise AssertionError("beam search should not use the cache scorer")

    generator = SequenceGenerator(
        score_fn=scorer.score_full,
        tokenizer=TinyTokenizer(),
        max_length=3,
        device="cpu",
        cache_score_fn=fail_cache_score,
    )

    sequences, scores = generator.generate_beam(
        torch.tensor([[0, 1]]),
        beam_width=2,
    )

    assert sequences.shape == (1, 2, 3)
    assert scores.shape == (1, 2)
    assert scorer.full_input_lengths == [2]


def test_beam_length_budget_and_finished_scores():
    prompt = torch.tensor([[0, 1]])

    def fail_score(_sequences):
        raise AssertionError("a zero-token beam budget must not call score_fn")

    no_budget = SequenceGenerator(
        score_fn=fail_score,
        tokenizer=TinyTokenizer(),
        max_length=prompt.size(1),
        device="cpu",
    )
    unchanged, unchanged_scores = no_budget.generate_beam(
        prompt.clone(),
        beam_width=2,
    )
    assert unchanged.shape == (1, 2, 2)
    assert torch.equal(unchanged[0, 0], prompt[0])
    assert torch.equal(unchanged[0, 1], prompt[0])
    assert torch.equal(unchanged_scores, torch.zeros_like(unchanged_scores))

    generator = SequenceGenerator(
        score_fn=BeamScorer.score_full,
        tokenizer=TinyTokenizer(),
        max_length=5,
        device="cpu",
    )
    sequences, scores = generator.generate_beam(prompt.clone(), beam_width=2)

    assert sequences.shape == (1, 2, 5)
    finished_index = (sequences[0, :, 2] == TinyTokenizer.eos_id).nonzero()[0, 0]
    assert sequences[0, finished_index].tolist() == [0, 1, 6, 6, 6]

    initial_log_probs = torch.log_softmax(
        BeamScorer.score_full(prompt),
        dim=-1,
    )
    expected_finished_score = initial_log_probs[0, TinyTokenizer.eos_id]
    assert torch.allclose(scores[0, finished_index], expected_finished_score)


def test_beam_argument_validation():
    scorer = HistoryScorer()
    prompt = torch.tensor([[0, 1]])
    generator = SequenceGenerator(
        score_fn=scorer.score_full,
        tokenizer=TinyTokenizer(),
        max_length=4,
        device="cpu",
    )

    invalid_arguments = [
        {"beam_width": 0},
        {"beam_width": 1.5},
        {"beam_width": TinyTokenizer.vocab_size + 1},
        {"beam_width": 2, "repeat_penalty": 0.0},
    ]
    for arguments in invalid_arguments:
        try:
            generator.generate_beam(prompt, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"beam search must reject {arguments}")


if __name__ == "__main__":
    test_beam_api_has_no_temperature()
    test_beam_uses_full_prefix_scorer()
    test_beam_length_budget_and_finished_scores()
    test_beam_argument_validation()
    print("All beam decoding tests passed.")
