from typing import Any, Callable, Optional, Tuple

import torch

from ..data import H4Tokenizer

'''
This file implements text generation strategies for transformer language models:
1. Greedy Search
2. Beam Search (Optional / Ungraded)
3. Sampling with Filtering (Temperature, Top-k, Top-p)
'''


class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            cache_score_fn: Optional[Callable] = None
    ) -> None:
        """
        Initialize the sequence generator.
        """
        # Add input validation (DO NOT MODIFY)
        if (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or max_length <= 0
        ):
            raise ValueError("max_length must be a positive integer")
        self.score_fn = score_fn
        self.cache_score_fn = cache_score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device(device)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Validate a prompt batch and move it to the configured device."""
        # Add input validation (DO NOT MODIFY)
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if x.size(1) == 0:
            raise ValueError("Input x must contain at least one token")
        prepared_input = x.to(self.device)
        return prepared_input

    def _score_next(
        self,
        sequences: torch.Tensor,
        cache_input: torch.Tensor,
        cache: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Score the next token using a full prefix or a cache."""
        # Select full-prefix or cached scoring (DO NOT MODIFY)
        if self.cache_score_fn is None:
            logits = self.score_fn(sequences)
            result = (logits, cache)
        else:
            result = self.cache_score_fn(cache_input, cache)
        return result

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """Apply repetition penalty to logits based on tokens in sequences."""
        # Apply the standard repetition penalty (DO NOT MODIFY)
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        '''
        Specification:
        - Apply temperature, top-k, and top-p filtering, in that order.
        - top-k only applies when top_k > 0.
        - top-p only applies when top_p < 1.0.
        - Filtering must mask tokens by setting their logit to float('-inf'),
          not by removing them from the tensor, so the shape never changes.
        '''
        # 1. Scale by temperature
        logits = logits / temperature

        # 2. Top-k filtering
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth_values = torch.topk(logits, k, dim=-1)[0][..., -1, None]
            logits = torch.where(logits < kth_values, torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype), logits)

        # 3. Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens with cumulative probability above top_p (shift right to keep the first token exceeding top_p)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter back to the original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation (DO NOT MODIFY)
        x = self._prepare_input(x)
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be > 0")

        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        cache = None
        cache_input = x

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            next_scores, cache = self._score_next(x, cache_input, cache)
            next_scores = self._apply_repeat_penalty(next_scores, x, repeat_penalty)

            log_probs = torch.log_softmax(next_scores, dim=-1)
            next_tokens = torch.argmax(next_scores, dim=-1)

            # Keep finished sequences on EOS
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.tokenizer.eos_id),
                next_tokens,
            )
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
            cache_input = next_tokens.unsqueeze(1)

            # Check if any sequence has reached EOS
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beam search is not required or graded."""
        x = self._prepare_input(x)
        if not isinstance(beam_width, int) or isinstance(beam_width, bool):
            raise ValueError("beam_width must be an integer")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if beam_width > self.tokenizer.vocab_size:
            raise ValueError("beam_width must not exceed the tokenizer vocabulary size")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be > 0")

        raise NotImplementedError

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repeat_penalty: float = 1.0,
            seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        """
        # Add input validation (DO NOT MODIFY)
        x = self._prepare_input(x)
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be > 0")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer")

        # Initialize the local random generator when a seed is provided (DO NOT MODIFY)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed)

        # Initialize scores, finished flags, and cache state (DO NOT MODIFY)
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        cache = None
        cache_input = x

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished (DO NOT MODIFY)
            if finished.all():
                break

            # Get logits and apply repeat penalty (DO NOT MODIFY)
            next_scores, cache = self._score_next(x, cache_input, cache) # (batch_size, vocab_size)
            next_scores = self._apply_repeat_penalty(
                next_scores,
                x,
                repeat_penalty,
            )

            # Filter logits and compute log-probabilities and probabilities
            filtered_logits = self._filter_logits(
                next_scores,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            probs = torch.softmax(filtered_logits, dim=-1)

            # Draw from the filtered probability distribution (DO NOT MODIFY)
            next_tokens = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)

            # Keep finished sequences on EOS (DO NOT MODIFY)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.tokenizer.eos_id),
                next_tokens,
            )
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences (DO NOT MODIFY)
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens (DO NOT MODIFY)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)
            cache_input = next_tokens.unsqueeze(1)

            # Check if any sequence has reached EOS (DO NOT MODIFY)
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(
            seq: torch.Tensor,
            tokenizer: H4Tokenizer,
            start_index: int = 0
    ) -> torch.Tensor | list[torch.Tensor]:
        """Post process sequences to remove content after EOS token."""
        # DO NOT MODIFY - this method is complete
        if seq.dim() not in (1, 2):
            raise ValueError("seq must be one- or two-dimensional")
        if not isinstance(start_index, int) or isinstance(start_index, bool):
            raise ValueError("start_index must be a non-negative integer")
        if start_index < 0 or start_index > seq.size(-1):
            raise ValueError("start_index must be within the sequence")

        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq[start_index:] == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = start_index + eos_indices[0].item() + 1
                processed = seq[:end_idx]
            else:
                processed = seq
            return processed

        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id
        eos_mask[:, :start_index] = False
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        processed = [s[:m.sum()] for s, m in zip(seq, seq_mask)]
        return processed