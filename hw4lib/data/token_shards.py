from collections.abc import Iterable
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

import numpy as np

from .tokenizer import H4Tokenizer
from .lm_dataset import FORMAT_VERSION, MANIFEST_FILENAME, SHARD_DTYPE

'''
Specification:
This preprocessing utility converts an ordered stream of documents into the
fixed GPT-2 token shards consumed by LMDataset.

Key Requirements:
- Tokenize each document without implicit special tokens
- Append exactly one GPT-2 end-of-text token after every document
- Store token IDs as little-endian uint16 values
- Split the token stream into bounded-size shards without reordering it
- Record corpus statistics and tokenizer metadata
- Stage all output before publishing the completed partition

Usage:
- Call once for each train/val/test document split
- Pass a new or empty output directory; existing data is never overwritten
'''


def write_token_shards(
    documents: Iterable[str],
    output_dir: str | Path,
    tokenizer: H4Tokenizer,
    shard_size: int,
) -> dict:
    """Tokenize documents and write a manifest plus raw uint16 shards.

    One GPT-2 end-of-text token is appended after every document. Existing
    files are never overwritten: ``output_dir`` must be new or empty.

    Args:
        documents: Text documents to tokenize in iteration order.
        output_dir: New or empty directory for the manifest and shards.
        tokenizer: ``H4Tokenizer`` configured with ``token_type='gpt2'``.
        shard_size: Maximum number of tokens in each shard.

    Returns:
        The manifest written to ``output_dir``.

    Raises:
        FileExistsError: If ``output_dir`` is not empty.
        TypeError: If a document is not a string.
        ValueError: If arguments or produced token IDs are invalid.
    """
    if (
        not isinstance(shard_size, int)
        or isinstance(shard_size, bool)
        or shard_size <= 0
    ):
        raise ValueError("shard_size must be a positive integer")
    if not 0 < tokenizer.vocab_size <= (1 << 16):
        raise ValueError("Tokenizer vocabulary does not fit in uint16")
    if tokenizer.token_type != "gpt2":
        raise ValueError("write_token_shards requires token_type='gpt2'")
    if not 0 <= tokenizer.eos_id < tokenizer.vocab_size:
        raise ValueError("Tokenizer eos_id must be inside the vocabulary")

    # Set up the destination without overwriting an existing partition.
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {destination}")

    # Stage the partition so a failed write leaves no partial destination.
    with TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}.tmp-",
    ) as staging_name:
        staging = Path(staging_name)
        shards: list[dict[str, int | str]] = []
        shard_file: BinaryIO | None = None
        shard_filename = ""
        shard_token_count = 0
        total_documents = 0
        total_chars = 0
        total_tokens = 0

        def open_shard() -> BinaryIO:
            """Open the next shard.

            Returns:
                Writable shard file.
            """
            nonlocal shard_filename
            shard_filename = f"shard_{len(shards):05d}.bin"
            path = staging / shard_filename
            file_handle = path.open("wb")
            return file_handle

        def close_shard() -> None:
            """Close and record the current shard.

            Returns:
                None.
            """
            nonlocal shard_file, shard_token_count
            if shard_file is None:
                return
            shard_file.close()
            shards.append(
                {
                    "file": shard_filename,
                    "num_tokens": shard_token_count,
                }
            )
            shard_file = None
            shard_token_count = 0

        try:
            for document in documents:
                if not isinstance(document, str):
                    raise TypeError("Every document must be a string")

                # Tokenize the document and add one explicit document boundary.
                token_ids = tokenizer.encode(document)
                token_ids.append(tokenizer.eos_id)
                tokens = np.asarray(token_ids, dtype=np.int64)
                if np.any(tokens < 0) or np.any(tokens >= tokenizer.vocab_size):
                    raise ValueError("A token ID is outside the tokenizer vocabulary")

                total_documents += 1
                total_chars += len(document)
                total_tokens += int(tokens.size)

                offset = 0
                while offset < tokens.size:
                    if shard_file is None:
                        shard_file = open_shard()

                    available = shard_size - shard_token_count
                    count = min(available, int(tokens.size) - offset)
                    chunk = tokens[offset : offset + count].astype(
                        SHARD_DTYPE, copy=False
                    )
                    chunk.tofile(shard_file)
                    shard_token_count += count
                    offset += count

                    if shard_token_count == shard_size:
                        close_shard()
        finally:
            close_shard()

        # Store everything required to reproduce and validate this partition.
        manifest = {
            "format_version": FORMAT_VERSION,
            "dtype": SHARD_DTYPE.str,
            "tokenizer": "gpt2",
            "vocab_size": tokenizer.vocab_size,
            "eos_id": tokenizer.eos_id,
            "num_documents": total_documents,
            "num_chars": total_chars,
            "num_tokens": total_tokens,
            "shards": shards,
        }
        manifest_path = staging / MANIFEST_FILENAME
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.write("\n")

        # Keep generated datasets readable while avoiding accidental execute or
        # write permissions from an unusually permissive process umask.
        for shard in shards:
            (staging / str(shard["file"])).chmod(0o644)
        manifest_path.chmod(0o644)
        staging.chmod(0o755)

        # Publish the completed staging directory as one final rename.
        if destination.exists():
            # This fails safely if another process populated the directory.
            destination.rmdir()
        staging.replace(destination)
        return manifest
