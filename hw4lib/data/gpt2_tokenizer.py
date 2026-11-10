from pathlib import Path

from .tokenizer import H4Tokenizer

'''
Specification:
This compatibility wrapper configures H4Tokenizer with the bundled GPT-2
byte-level BPE vocabulary.

Key Requirements:
- Use <|endoftext|> as the document boundary/end token
- Do not append that token implicitly in encode()
- Keep the vocabulary small enough for the uint16 shard format
- Preserve a no-argument constructor for the standard bundled tokenizer

Usage:
- GPT2Tokenizer() loads the homework's canonical GPT-2 tokenizer JSON
- GPT2Tokenizer(path) is useful for small offline tests and custom JSON files
'''


class GPT2Tokenizer(H4Tokenizer):
    """
    Backward-compatible GPT-2 tokenizer setup for language modeling.

    This wrapper preserves the earlier no-argument setup while all tokenizer
    behavior lives in ``H4Tokenizer``.
    """

    EOS_TOKEN = H4Tokenizer.GPT2_EOS_TOKEN
    UINT16_VOCAB_LIMIT = 1 << 16
    DEFAULT_TOKENIZER_FILE = (
        Path(__file__).parent / "tokenizer_jsons" / "tokenizer_gpt2.json"
    )

    def __init__(self, tokenizer_file: str | Path | None = None) -> None:
        """Load a GPT-2 tokenizer JSON.

        Args:
            tokenizer_file: Optional tokenizer JSON path. The bundled GPT-2
                file is used when this value is ``None``.

        Raises:
            FileNotFoundError: If the tokenizer JSON does not exist.
            ValueError: If the JSON is invalid or does not fit uint16 shards.
        """
        # Use the canonical bundled vocabulary unless a test or experiment
        # supplies another GPT-2-compatible tokenizer JSON.
        tokenizer_path = Path(tokenizer_file or self.DEFAULT_TOKENIZER_FILE)
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        # Delegate tokenization behavior to the shared homework interface.
        try:
            super().__init__(
                token_map={"gpt2": tokenizer_path},
                token_type="gpt2",
                validate=False,
            )
        except ValueError as exc:
            if "missing its end token" in str(exc):
                raise ValueError(
                    f"Tokenizer is missing the GPT-2 end token {self.EOS_TOKEN!r}"
                ) from exc
            raise ValueError(
                f"Could not load tokenizer JSON: {tokenizer_path}"
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"Could not load tokenizer JSON: {tokenizer_path}"
            ) from exc

        # Token shards store IDs as uint16, so validate the complete tokenizer
        # contract once at construction time.
        if not 0 <= self.eos_id < self.vocab_size:
            raise ValueError("Tokenizer end-of-text ID is outside its vocabulary")
        if self.vocab_size > self.UINT16_VOCAB_LIMIT:
            raise ValueError(
                "Tokenizer vocabulary does not fit in the uint16 shard format: "
                f"{self.vocab_size} tokens"
            )
