from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import List, Literal

from tokenizers import Tokenizer, decoders, processors

TokenType = Literal['char', '1k', '5k', '10k', 'gpt2']

'''
Specification:
You will be splitting each text transcript into tokenized text, which will then serve as inputs to your model. 
Additionally, you will need a method to convert the tokenized text back to its original form during inference or generation. 
Character-level tokenization is straightforward, while subword tokenization involves splitting a word into subwords rather than characters. 
The specific algorithm used for subword tokenization is [Byte Pair Encoding (BPE)](https://arxiv.org/pdf/1508.07909). 
Part of the assignment will involve exploring these different tokenization strategies to assess their impact on your model's performance.

The H4Tokenizer class provides tokenization functionality for the homework.

1. Tokenizer Types:
   - Supports multiple tokenization strategies:
     * char: Character-level tokenization
     * 1k: Subword tokenization with 1000 vocab size
     * 5k: Subword tokenization with 5000 vocab size
     * 10k: Subword tokenization with 10000 vocab size

2. Special Tokens:
   - [PAD]: Used for padding sequences to same length
   - [UNK]: Represents unknown/out-of-vocabulary tokens
   - [MASK]: Used for masked language modeling
   - [SOS]: Marks start of sequence
   - [EOS]: Marks end of sequence
   - [BLANK]: Used for blank tokens in ASR

3. Key Operations:
   - encode(): Converts text to token IDs
   - decode(): Converts token IDs back to text
   - tokenize(): Splits text into tokens without converting to IDs
   - get_avg_chars_per_token(): Calculates character/token ratio for perplexity

4. Validation:
   - Automatically validates tokenizer at initialization
   - Shows example tokenization/detokenization
   - Displays configuration and special token IDs

Usage:
- Used by LMDataset to process text data
- Ensures consistent tokenization across training/validation
- Handles special token addition and sequence conversion

HW4P1 Extension:
- gpt2 uses GPT-2 byte-level BPE for fixed-window language modeling
- <|endoftext|> marks GPT-2 document boundaries
- Document-boundary tokens are added by preprocessing rather than encode()
'''


class H4Tokenizer:
    """
    A tokenizer class that supports character-level and subword tokenization strategies.
    Loads pre-trained tokenizers and provides methods for encoding/decoding text.

    Attributes:
        token_type (str): Type of tokenizer ('char', '1k', '5k', '10k', 'gpt2')
        tokenizer (Tokenizer): The underlying tokenizer object
        pad_id (int | None): ID for padding token '[PAD]'
        unk_id (int | None): ID for unknown token '[UNK]'
        mask_id (int | None): ID for mask token '[MASK]'
        sos_id (int | None): ID for start-of-sequence token '[SOS]'
        eos_id (int): ID for the tokenizer's end token
        blank_id (int | None): ID for blank token '[BLANK]'
    """

    VALID_TYPES = ['char', '1k', '5k', '10k', 'gpt2']
    GPT2_EOS_TOKEN = "<|endoftext|>"

    def __init__(self, token_map: Mapping[str, str | Path], token_type: TokenType="char", validate: bool=True) -> None:
        """
        Initialize tokenizer from pre-trained file.

        Args:
            token_map: Maps token types to tokenizer file paths
            token_type: Type of tokenizer to load
            validate: Whether to validate the tokenizer
        
        Raises:
            FileNotFoundError: If the tokenizer JSON does not exist.
            KeyError: If ``token_map`` has no path for ``token_type``.
            ValueError: If ``token_type`` is invalid or has no end token.
        """
        if token_type not in self.VALID_TYPES:
            raise ValueError(f"token_type must be one of {self.VALID_TYPES}")
        if token_type not in token_map:
            raise KeyError(f"token_map has no path for token type {token_type!r}")

        self.token_type = token_type
        tokenizer_path = Path(token_map[token_type])
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        self.tokenizer_path = tokenizer_path
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Configure decoder based on tokenizer type
        if token_type == "char":
            self.tokenizer.decoder = decoders.Fuse()
        elif token_type != "gpt2":
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
            self.tokenizer.decoder = decoders.ByteLevel()

        # Get vocabulary size
        self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)

        # Load special token IDs
        self.pad_id   = self.tokenizer.token_to_id("[PAD]")
        self.unk_id   = self.tokenizer.token_to_id("[UNK]")
        self.mask_id  = self.tokenizer.token_to_id("[MASK]")
        self.sos_id   = self.tokenizer.token_to_id("[SOS]")
        self.blank_id = self.tokenizer.token_to_id("[BLANK]")
        eos_token = self.GPT2_EOS_TOKEN if token_type == "gpt2" else "[EOS]"
        self.eos_id = self.tokenizer.token_to_id(eos_token)
        if self.eos_id is None:
            raise ValueError(f"Tokenizer is missing its end token {eos_token!r}")

        if validate:
            self._validate_tokenizer()

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of token strings
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = encoding.tokens
        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = encoding.ids
        return token_ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool=False) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to omit registered special tokens

        Returns:
            Decoded text string
        """
        ids = [int(token_id) for token_id in token_ids]
        text = self.tokenizer.decode(
            ids,
            skip_special_tokens=skip_special_tokens,
        )
        return text

    def _validate_tokenizer(self) -> None:
        """
        Validate tokenizer functionality by testing basic operations.
        Prints diagnostic information and test results.
        """
        if self.token_type == "gpt2":
            test_text = "Hello, deep learners!"
        else:
            test_text = "[SOS]HI DEEP LEARNERS[EOS]"
        tokens    = self.tokenize(test_text)
        ids       = self.encode(test_text)
        decoded   = self.decode(ids)

        print("="*80)
        title = f"Tokenizer Configuration ({self.token_type})"
        print(f"{title:^80}")
        print("-"*80)
        print(f"{'Vocabulary size:':<20} {self.tokenizer.get_vocab_size()}")
        print("\nSpecial Tokens:")
        print(f"{'PAD:':<12} {str(self.pad_id):>6}")
        print(f"{'UNK:':<12} {str(self.unk_id):>6}")
        print(f"{'MASK:':<12} {str(self.mask_id):>6}")
        print(f"{'SOS:':<12} {str(self.sos_id):>6}")
        print(f"{'EOS:':<12} {str(self.eos_id):>6}")
        print(f"{'BLANK:':<12} {str(self.blank_id):>6}")
        print("\nValidation Example:")
        print("-"*80)
        print(f"{'Input text:':<12} {test_text}")
        print(f"{'Tokens:':<12} {tokens}")
        print(f"{'Token IDs:':<12} {ids}")
        print(f"{'Decoded:':<12} {decoded}")
        print("="*80)

    def get_avg_chars_per_token(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> float:
        """
        Calculate average number of characters per token for given token IDs.

        Args:
            token_ids: List of token IDs to analyze
            skip_special_tokens: Whether to exclude special tokens from calculation

        Returns:
            Average number of characters per token
        """
        decoded_text = self.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if skip_special_tokens:
            token_count = sum(1 for id in token_ids if id not in
                              [self.pad_id, self.unk_id, self.mask_id,
                               self.sos_id, self.eos_id, self.blank_id])
        else:
            token_count = len(token_ids)

        average = len(decoded_text) / token_count if token_count > 0 else 0.0
        return average
