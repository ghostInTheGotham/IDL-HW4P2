import torch

from hw4lib.decoding.sequence_generator import SequenceGenerator


class TinyTokenizer:
    """Tokenizer metadata used by the decoding tests."""

    vocab_size = 7
    eos_id = 6


class HistoryScorer:
    """Score the next token from a small deterministic transition table."""

    def __init__(self):
        self.full_input_lengths = []
        self.cached_input_lengths = []

    @staticmethod
    def logits(history: torch.Tensor) -> torch.Tensor:
        transitions = torch.tensor(
            [6, 3, 4, 6, 5, 6, 1],
            device=history.device,
        )
        next_tokens = transitions[history[:, -1]]
        logits = torch.full(
            (history.size(0), TinyTokenizer.vocab_size),
            -4.0,
            device=history.device,
        )
        logits.scatter_(1, next_tokens.unsqueeze(1), 8.0)
        return logits

    def score_full(self, sequences: torch.Tensor) -> torch.Tensor:
        self.full_input_lengths.append(sequences.size(1))
        return self.logits(sequences)

    def score_cached(self, new_tokens: torch.Tensor, cache):
        self.cached_input_lengths.append(new_tokens.size(1))
        if cache is None:
            history = new_tokens.clone()
        else:
            history = torch.cat([cache, new_tokens], dim=1)
        return self.logits(history), history


class BeamScorer:
    """Keep one beam running while another finishes immediately."""

    @staticmethod
    def score_full(sequences: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (sequences.size(0), TinyTokenizer.vocab_size),
            -20.0,
            device=sequences.device,
        )
        for row, last_token in enumerate(sequences[:, -1].tolist()):
            if last_token == 1:
                logits[row, 3] = 5.0
                logits[row, TinyTokenizer.eos_id] = 4.0
            elif last_token == 3:
                logits[row, 4] = 8.0
            elif last_token == 4:
                logits[row, 5] = 8.0
            else:
                logits[row, 0] = 8.0
        return logits


def make_generator(scorer, max_length: int, use_cache: bool) -> SequenceGenerator:
    return SequenceGenerator(
        score_fn=scorer.score_full,
        tokenizer=TinyTokenizer(),
        max_length=max_length,
        device="cpu",
        cache_score_fn=scorer.score_cached if use_cache else None,
    )


def repeated_token_logits(sequences: torch.Tensor) -> torch.Tensor:
    """Prefer a repeated token unless repetition penalty is applied."""

    logits = torch.zeros(
        sequences.size(0),
        TinyTokenizer.vocab_size,
        device=sequences.device,
    )
    logits[:, 1] = 4.0
    logits[:, 2] = 3.0
    return logits


def sampling_logits(sequences: torch.Tensor) -> torch.Tensor:
    """Give four non-EOS tokens equal sampling probability."""

    logits = torch.full(
        (sequences.size(0), TinyTokenizer.vocab_size),
        float("-inf"),
        device=sequences.device,
    )
    logits[:, :4] = 0.0
    return logits
