import torch.nn as nn
import torch 
from typing import Tuple, Optional
from .causal_self_attention import CausalSelfAttention, KVCache

'''
The file contains two sublayers used in transformer decoders:
1. SelfAttentionLayer: For masked self-attention
2. FeedForwardLayer: For position-wise feed-forward processing

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Normalization is applied before the main operation
- A residual connection wraps around the operation
'''

class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    This layer is responsible for the causally-masked self-attention mechanism.
    ''' 
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        use_rope: bool = False,
        max_len: int = 2048,
        num_kv_heads: Optional[int] = None,
        rope_base: float = 10_000.0,
    ):
        '''
        Initialize the SelfAttentionLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            use_rope (bool): Whether to use rotary causal attention.
            max_len (int): The maximum context length for rotary attention.
            num_kv_heads (Optional[int]): The number of key/value heads.
            rope_base (float): The frequency base for rotary positions.
        '''
        super().__init__()

        # DO NOT MODIFY
        # Only the rotary path has key/value groups
        self.use_rope = use_rope
        if not use_rope and num_kv_heads not in (None, num_heads):
            raise ValueError("num_kv_heads requires use_rope=True")

        if use_rope:
            # Initialize rotary causal self-attention
            self.mha = CausalSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                max_len=max_len,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                rope_base=rope_base,
            )
        else:
            # DO NOT MODIFY
            # This is the additive-positional path used by HW4P2 and is not
            # part of the HW4P1 exercise
            self.mha = nn.MultiheadAttention(
                embed_dim   = d_model,
                num_heads   = num_heads,
                dropout     = dropout,
                batch_first = True,
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the SelfAttentionLayer.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model)   
            key_padding_mask (Optional[torch.Tensor]): Padding mask, shape (batch_size, seq_len)
            attn_mask (Optional[torch.Tensor]): Attention mask, shape (seq_len, seq_len)

        Returns:
            x (torch.Tensor): Output tensor, shape (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): Attention weights, shape (batch_size, seq_len, seq_len)   
        '''
        residual = x
        normalized = self.norm(x)

        if self.use_rope:
            # Rotary attention returns: output, attention_weights, present_kv
            attn_output, mha_attn_weights, _ = self.mha(
                normalized,
                past_kv=None,
                use_cache=False,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            # DO NOT MODIFY
            # Additive-positional path
            attn_output, mha_attn_weights = self.mha(
                query                = normalized,
                key                  = normalized,
                value                = normalized,
                key_padding_mask     = key_padding_mask,
                need_weights         = True,
                attn_mask            = attn_mask,
                average_attn_weights = True,
            )

        x = residual + self.dropout(attn_output)
        return x, mha_attn_weights

    def forward_cached(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, KVCache]:
        '''
        Apply Pre-LN rotary self-attention while extending a compact
        key/value cache.
        '''
        # DO NOT MODIFY: KV caching is available only on the rotary path.
        if not self.use_rope:
            raise ValueError("KV caching requires use_rope=True")

        # Save the residual and apply pre-normalization
        residual = x
        normalized = self.norm(x)

        # Call self.mha with past_kv and use_cache=True
        attention_output, mha_attn_weights, present_kv = self.mha(
            normalized,
            past_kv=past_kv,
            use_cache=True,
        )

        # Apply dropout and the residual connection
        x = residual + self.dropout(attention_output)

        # DO NOT MODIFY: a cached attention call must return cache state.
        if present_kv is None:
            raise RuntimeError("Cached attention did not return a key/value cache")

        return x, mha_attn_weights, present_kv


class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    This layer is responsible for the position-wise feed-forward network.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        '''
        Initialize the FeedForwardLayer. 
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the FeedForwardLayer.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model)   

        Returns:
            x (torch.Tensor): Output tensor, shape (batch_size, seq_len, d_model)
        '''
        residual = x
        normalized = self.norm(x)
        out = self.ffn(normalized)
        x = residual + self.dropout(out)
        return x