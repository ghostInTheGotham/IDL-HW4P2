import copy
import math
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from tests.test_sft_dataset import TinyTokenizer, records


class ToyLM(nn.Module):
    num_classes = 16
    max_len = 16
    use_rope = False

    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(16, 16)
        self.seen = []

    def forward(self, ids):
        return self.table(ids), {}

    def score(self, ids):
        self.seen.append(ids.tolist())
        return self.table(ids[:, -1])


def trainer(tmp_path, model=None, name='run', accumulation=1, resume=False):
    from hw4lib.trainers.sft_trainer import SFTTrainer
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('test: true\n')
    config = {'output_dir': str(tmp_path / 'runs'), 'training': {'use_wandb': False, 'epochs': 1,
              'gradient_accumulation_steps': accumulation, 'gradient_clip': 100, 'amp': False, 'resume': resume},
              'generation': {'max_new_tokens': 4}}
    t = SFTTrainer(model or ToyLM(), TinyTokenizer(), config, name, str(config_file), device='cpu')
    t.optimizer = torch.optim.SGD(t.model.parameters(), lr=.2)
    return t


def loader(batch_size):
    from hw4lib.data.sft_dataset import SFTDataset
    ds = SFTDataset(records(), TinyTokenizer(), 16)
    return DataLoader(ds, batch_size=batch_size, collate_fn=ds.collate_fn)


def test_accumulation_matches_combined_batch_including_partial_cycle(tmp_path):
    a = trainer(tmp_path, name='a', accumulation=3)
    b = trainer(tmp_path, model=copy.deepcopy(a.model), name='b')
    a._train_epoch(loader(1))
    b._train_epoch(loader(2))
    torch.testing.assert_close(a.model.table.weight, b.model.table.weight)
    assert a.global_optimizer_step == b.global_optimizer_step == 1


def test_response_loss_and_small_learning_run(tmp_path):
    t = trainer(tmp_path)
    before = t._validate_epoch(loader(2))
    for _ in range(15): t._train_epoch(loader(2))
    after = t._validate_epoch(loader(1))
    assert after['response_ce'] < before['response_ce']
    assert after['response_tokens'] == 5
    assert after['response_ppl'] == pytest.approx(math.exp(after['response_ce']))


def test_prompt_only_generation_stops_at_eos(tmp_path):
    t = trainer(tmp_path)
    with torch.no_grad():
        t.model.table.weight.zero_()
        t.model.table.weight[3, 1] = 20
    assert t.generate(records()) == ['']
    assert t.model.seen == [[[2, 3]]]


def test_resumed_evaluation_saves_current_predictions_and_preserves_old_outputs(tmp_path):
    from hw4lib.utils.sft_experiment import evaluate_and_export
    import json
    t = trainer(tmp_path)
    t.config['training']['resume'] = True
    (t.expt_root / 'metrics.json').write_text(json.dumps({'response_ce': 999}))
    (t.expt_root / 'dev_predictions.csv').write_text('id,prediction\na,stale answer\n')
    with torch.no_grad():
        t.model.table.weight.zero_()
        t.model.table.weight[3, 1] = 20
    metrics, predictions = evaluate_and_export(t, loader(2), records(), available_inputs=1, seed=11785)
    assert predictions == ['']
    assert metrics['response_tokens'] == 5
    previous = next(t.expt_root.glob('evaluation-history/*/metrics.json'))
    assert json.loads(previous.read_text()) == {'response_ce': 999}


def test_checkpoint_resume(tmp_path):
    t = trainer(tmp_path)
    t._train_epoch(loader(2))
    t.current_epoch = 1
    t.save_checkpoint('last.pth')
    resumed = trainer(tmp_path, resume=True)
    resumed.load_checkpoint('last.pth')
    assert resumed.current_epoch == 1
    assert resumed.global_optimizer_step == 1
    t._train_epoch(loader(2))
    resumed._train_epoch(loader(2))
    torch.testing.assert_close(t.model.table.weight, resumed.model.table.weight)


def test_best_checkpoint_uses_combined_kaggle_score(tmp_path, monkeypatch):
    t = trainer(tmp_path)
    t.config['training']['epochs'] = 2
    scores = iter([
        {'response_ce': 1., 'response_ppl': math.e, 'response_tokens': 5,
         'bleu': 90., 'rouge_l': 10., 'kaggle_score': 50., 'examples': 1},
        {'response_ce': 1., 'response_ppl': math.e, 'response_tokens': 5,
         'bleu': 60., 'rouge_l': 80., 'kaggle_score': 70., 'examples': 1},
    ])
    monkeypatch.setattr(t, 'evaluate', lambda *args: next(scores))
    t.train(loader(2), loader(2), records())
    best = torch.load(t.checkpoint_dir / 'best.pth', weights_only=True)
    assert best['epoch'] == 2
    assert best['best_metric'] == pytest.approx(70)





def test_right_padding_invariance():
    from hw4lib.model import DecoderOnlyTransformer
    model = DecoderOnlyTransformer(1, 8, 2, 16, 0, 16, 16, num_kv_heads=1).eval()
    with torch.no_grad():
        short = model(torch.tensor([[2, 3, 4]]))[0]
        padded = model(torch.tensor([[2, 3, 4, 1, 1]]))[0]
    torch.testing.assert_close(short, padded[:, :3], atol=1e-6, rtol=1e-5)


