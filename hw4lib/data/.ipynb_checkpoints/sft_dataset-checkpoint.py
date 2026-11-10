"""Supervised prompt/response examples for HW4P2."""
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from ..utils.webnlg import format_webnlg_prompt


class SFTDataset(Dataset):
    """Build response-only next-token targets from grouped WebNLG records.

    Args:
        records: Inputs with id, triples, and a list of references.
        tokenizer: P1 tokenizer with encode() and eos_id.
        max_length: Maximum number of input tokens, excluding the shifted EOS.
        expand_references: Train on all references; use the first for dev CE.
    """
    def __init__(self, records, tokenizer, max_length, *, expand_references=True):
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise ValueError("max_length must be positive")
        if not records:
            raise ValueError("The dataset must contain examples")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for record in records:
            refs = record.get("references")
            if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or not r.strip() for r in refs):
                raise ValueError("Every labeled input needs nonempty references")
            prompt = tokenizer.encode(format_webnlg_prompt(record["triples"]))
            for response in refs if expand_references else refs[:1]:
                answer = tokenizer.encode(response)
                if not prompt or not answer:
                    raise ValueError("Prompt and response must contain tokens")
                if len(prompt) + len(answer) > max_length:
                    raise ValueError(f"Input {record['id']} exceeds context {max_length}; no text was truncated")
                self.examples.append(self._make_example(prompt, answer))

    def _make_example(self, prompt, answer):
        """Return shifted input_ids and labels; supervise response and EOS only."""
        # TODO: Join prompt, response, and EOS. Shift once for next-token prediction.
        # The first response target sits at index len(prompt) - 1.
        raise NotImplementedError("Complete _make_example")

    def __len__(self):
        # TODO: Return the number of prompt/response pairs.
        raise NotImplementedError("Complete __len__")

    def __getitem__(self, index):
        # TODO: Return one example without allowing the caller to alter stored tensors.
        raise NotImplementedError("Complete __getitem__")

    def collate_fn(self, examples):
        """Right-pad inputs with EOS, and labels with -100 (not EOS)."""
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        # TODO: Pad each field to the longest example in this batch.
        raise NotImplementedError("Complete collate_fn")
