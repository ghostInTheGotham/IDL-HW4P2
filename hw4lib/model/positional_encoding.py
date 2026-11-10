import torch
from torch import nn
import math

'''
TODO: Implement these Modules.

This file contains two ways of giving a transformer positional information:

1. PositionalEncoding: Fixed sinusoidal encoding added to the token embeddings
   - Added once, before the first layer
   - Encodes absolute position

2. RotaryPositionalEncoding: Rotates queries and keys inside each attention operation
   - Applied per head, after the q/k projections and the reshape into heads
   - Encodes relative position

Specification for RotaryPositionalEncoding:
- Applied to queries and keys only, never to values
- Pairs dimension i with dimension i + head_dim//2 (the "split-half" convention
  used by Llama and HuggingFace), not with dimension i+1
- The angle for position t and pair i is t / (base ** (2i / head_dim))
- cos and sin are registered as buffers of shape (1, 1, max_len, head_dim),
  so that they broadcast over batch and head
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # Register as buffer to save with model state
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length 
        """
        # Step 1: Get sequence length from input tensor
        seq_len = x.size(1)
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        # Step 3: Add positional encodings to input
        return x + self.pe[:, :seq_len, :]

## -------------------------------------------------------------------------------------------------

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).
    Rotates queries and keys so that attention scores depend on relative position.
    """
    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, but got {head_dim}")
        if max_len <= 0:
            raise ValueError(f"max_len must be greater than 0, got {max_len}")

        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base
        self.create_rope_table(head_dim, max_len, base)

    def create_rope_table(self, head_dim: int, max_len: int, base: float):
        # Inverse frequencies for head_dim // 2 frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)  # (max_len, head_dim // 2)

        # Interleave frequencies to match neighboring pairs (0, 1), (2, 3), ...
        freqs = freqs.repeat_interleave(2, dim=-1)  # (max_len, head_dim)

        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, max_len, head_dim)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, max_len, head_dim)

        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates neighboring pairs: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
        """
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x0 = x_reshaped[..., 0]
        x1 = x_reshaped[..., 1]
        rotated = torch.stack((-x1, x0), dim=-1)
        return rotated.view_as(x)

    def _apply_rope(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor of shape (batch, num_heads, seq_len, head_dim), got shape {x.shape}")
        
        seq_len = x.size(2)
        if start_pos + seq_len > self.max_len:
            raise ValueError(f"Sequence length {start_pos + seq_len} exceeds max_len {self.max_len}")

        cos_val = self.cos[:, :, start_pos:start_pos + seq_len, :]
        sin_val = self.sin[:, :, start_pos:start_pos + seq_len, :]
        return (x * cos_val) + (self.rotate_half(x) * sin_val)

    def forward(self, x: torch.Tensor, k: torch.Tensor = None, start_pos: int = 0, offset: int = None):
        if offset is not None:
            start_pos = offset

        if k is None:
            return self._apply_rope(x, start_pos)
        else:
            return self._apply_rope(x, start_pos), self._apply_rope(k, start_pos)