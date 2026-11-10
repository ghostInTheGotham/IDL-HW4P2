"""Low-rank adaptation of P1 attention projections."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear projection plus (alpha / rank) B A dropout(x).

    A has shape (rank, in_features), B has shape (out_features, rank).
    Initialize B to zero so inserting LoRA leaves predictions unchanged.
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Require positive alpha and 0 <= dropout < 1")
        self.base, self.rank, self.alpha = base, rank, float(alpha)
        self.dropout = nn.Dropout(dropout)
        
        # Freeze the original linear projection
        for param in self.base.parameters():
            param.requires_grad = False
            
        # Create trainable A and B on the base weight's device and dtype
        # A has shape (rank, in_features)
        self.lora_A = nn.Parameter(self.base.weight.new_empty((rank, self.base.in_features)))
        
        # B has shape (out_features, rank)
        self.lora_B = nn.Parameter(self.base.weight.new_zeros((self.base.out_features, rank)))
        
        # Initialize A with Kaiming uniform, and B is already zeroed
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Compute the frozen base projection
        base_output = self.base(x)
        
        # Apply dropout only to the LoRA branch
        dropped_x = self.dropout(x)
        
        # Project from in_features to rank using A
        low_rank_features = F.linear(dropped_x, self.lora_A)
        
        # Project from rank to out_features using B
        lora_update = F.linear(low_rank_features, self.lora_B)
        
        # Scale the low-rank update and add to base output
        scale = self.alpha / self.rank
        return base_output + scale * lora_update


def inject_lora(model, *, rank, alpha, dropout, targets=("q_proj", "v_proj")):
    """Replace named projection leaves, then freeze all non-adapter parameters.

    Return the replaced module paths. Validate before mutating the model.
    """
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("LoRA is already installed")
        
    # Validate the target specification
    if isinstance(targets, str) or not targets or any(not isinstance(t, str) or not t for t in targets):
        raise ValueError("targets must be a nonempty sequence of projection names")
        
    # Find nn.Linear leaves whose names match targets
    matches = []
    for name, module in model.named_modules():
        leaf_name = name.split('.')[-1]
        if isinstance(module, nn.Linear) and leaf_name in targets:
            matches.append((name, module))
            
    # Reject any unmatched target
    matched_targets = {name.split('.')[-1] for name, _ in matches}
    missing_targets = set(targets) - matched_targets
    if missing_targets:
        raise ValueError(f"No linear projections found for requested targets: {missing_targets}")

    # Build replacements before modifying the model
    replacements = []
    for name, module in matches:
        replacement = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        replacements.append((name, replacement))
        
    # Freeze every parameter in the original model
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace modules and enable gradients for adapter parameters
    replaced_paths = []
    for name, replacement in replacements:
        parts = name.split('.')
        parent_path = '.'.join(parts[:-1])
        leaf_name = parts[-1]
        
        if not parent_path:
            parent = model
        else:
            parent = model.get_submodule(parent_path)
            
        setattr(parent, leaf_name, replacement)
        
        replacement.lora_A.requires_grad = True
        replacement.lora_B.requires_grad = True
        
        replaced_paths.append(name)
        
    return replaced_paths


def _adapter_config(model):
    return {name: {"rank": module.rank, "alpha": module.alpha,
                   "dropout": module.dropout.p,
                   "in_features": module.base.in_features,
                   "out_features": module.base.out_features}
            for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def save_adapter(model, path: Path, metadata: dict, *, reuse=False) -> None:
    """Save current adapter tensors; reuse=True retains the previous file in history."""
    from time import time_ns
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if name.endswith((".lora_A", ".lora_B"))}
    if reuse and Path(path).exists():
        history = Path(path).parent / "adapter-history" / str(time_ns())
        history.mkdir(parents=True)
        Path(path).replace(history / Path(path).name)
    with Path(path).open("xb") as file:
        torch.save({
            "submission_tracking": {
                "version": 1,
                "wandb_ref": getattr(
                    model.module if hasattr(model, "module") else model,
                    "_submission_wandb_ref",
                    None,
                ),
            },
            "config": _adapter_config(model),
            "state_dict": state,
            "metadata": metadata
        }, file)


def load_adapter(model, path: Path, *, base_checkpoint=None) -> dict:
    """Load adapter tensors onto the P1 model with LoRA installed."""
    tracked_model = model.module if hasattr(model, "module") else model
    tracked_model._submission_wandb_ref = None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith((".lora_A", ".lora_B")):
                parameter.copy_(payload["state_dict"][name].to(parameter))
    tracking = payload.get("submission_tracking", {})
    tracked_model._submission_wandb_ref = (
        tracking.get("wandb_ref")
        if isinstance(tracking, dict)
        else None
    )
    return payload.get("metadata", {})