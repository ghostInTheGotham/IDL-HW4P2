import torch


class TinyTokenizer:
    eos_id = 1
    pad_id = 1
    vocab_size = 16

    def encode(self, text):
        return [2, 3] if text.endswith('Description:') else [4, 5] if text == 'two tokens' else [6]

    def decode(self, ids, **kwargs):
        return ' '.join(str(i) for i in ids if i != self.eos_id)


def records():
    return [{'id': 'a', 'triples': ['A | p | B'], 'references': ['two tokens', 'short']}]


def test_shift_prompt_mask_and_eos():
    from hw4lib.data.sft_dataset import SFTDataset
    ds = SFTDataset(records(), TinyTokenizer(), 10)
    assert len(ds) == 2
    assert ds[0]['input_ids'].tolist() == [2, 3, 4, 5]
    assert ds[0]['labels'].tolist() == [-100, 4, 5, 1]
    assert len(SFTDataset(records(), TinyTokenizer(), 10, expand_references=False)) == 1


def test_collate_right_padding_does_not_mask_real_eos():
    from hw4lib.data.sft_dataset import SFTDataset
    ds = SFTDataset(records(), TinyTokenizer(), 10)
    batch = ds.collate_fn([ds[0], ds[1]])
    assert batch['input_ids'].tolist() == [[2, 3, 4, 5], [2, 3, 6, 1]]
    assert batch['labels'].tolist() == [[-100, 4, 5, 1], [-100, 6, 1, -100]]
    assert batch['labels'].dtype == torch.long


