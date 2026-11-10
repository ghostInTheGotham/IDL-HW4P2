import copy
import torch
from torch import nn


def model():
    return nn.ModuleDict({'q_proj': nn.Linear(4, 4, bias=False), 'v_proj': nn.Linear(4, 2), 'head': nn.Linear(4, 3)})


def test_initial_equivalence_and_known_update():
    from hw4lib.model.lora import LoRALinear
    torch.manual_seed(5)
    base = nn.Linear(4, 2)
    layer = LoRALinear(base, 2, 4, 0)
    x = torch.ones(3, 4)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    with torch.no_grad():
        layer.lora_A.fill_(1)
        layer.lora_B.fill_(1)
    torch.testing.assert_close(layer(x), base(x) + 16)


def test_freezing_and_gradients():
    from hw4lib.model.lora import inject_lora
    m = model()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    assert sorted(inject_lora(m, rank=2, alpha=4, dropout=0)) == ['q_proj', 'v_proj']
    optimizer = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.1)
    for _ in range(2):
        optimizer.zero_grad()
        m['q_proj'](torch.ones(2, 4)).sum().backward()
        optimizer.step()
    assert m['q_proj'].lora_A.grad.abs().sum() > 0
    assert m['q_proj'].lora_B.grad.abs().sum() > 0
    for name, p in m.named_parameters():
        assert p.requires_grad == (name.endswith('lora_A') or name.endswith('lora_B'))
    for name, expected in before.items():
        actual = m.state_dict()[name.replace('.weight', '.base.weight').replace('.bias', '.base.bias') if not name.startswith('head.') else name]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adapter_roundtrip_after_moving_files(tmp_path):
    from hw4lib.model.lora import inject_lora, save_adapter, load_adapter
    m = model()
    other = copy.deepcopy(m)
    for obj in (m, other): inject_lora(obj, rank=2, alpha=4, dropout=0)
    with torch.no_grad(): m['q_proj'].lora_B.fill_(.2)
    file = tmp_path / 'adapter.pth'
    save_adapter(m, file, {'base_checkpoint': 'base.pth'})
    load_adapter(other, file)
    x = torch.randn(2, 4)
    torch.testing.assert_close(m['q_proj'](x), other['q_proj'](x))
    assert all('lora_' in k for k in torch.load(file, weights_only=True)['state_dict'])
    previous = m['q_proj'](x).detach().clone()
    with torch.no_grad(): m['q_proj'].lora_B.add_(1)
    save_adapter(m, file, {'base_checkpoint': 'base.pth'}, reuse=True)
    load_adapter(other, file)
    torch.testing.assert_close(m['q_proj'](x), other['q_proj'](x))
    saved = next(tmp_path.glob('adapter-history/*/adapter.pth'))
    load_adapter(other, saved)
    torch.testing.assert_close(other['q_proj'](x), previous)


def test_adapter_roundtrip_preserves_metadata_and_tracking(tmp_path):
    from hw4lib.model.lora import inject_lora, save_adapter, load_adapter
    original = model()
    restored = copy.deepcopy(original)
    tracking = {'entity': 'qa', 'project': 'hw4p2', 'run_id': 'full-to-lora', 'run_name': 'adapter'}
    original._submission_wandb_ref = tracking
    for item in (original, restored):
        inject_lora(item, rank=2, alpha=4, dropout=0)
    metadata = {'base_checkpoint': 'base.pth', 'selected_checkpoint': {'epoch': 1, 'step': 3}}
    path = tmp_path / 'adapter.pth'
    save_adapter(original, path, metadata)
    assert load_adapter(restored, path) == metadata
    assert restored._submission_wandb_ref == tracking
    assert metadata == {'base_checkpoint': 'base.pth', 'selected_checkpoint': {'epoch': 1, 'step': 3}}




