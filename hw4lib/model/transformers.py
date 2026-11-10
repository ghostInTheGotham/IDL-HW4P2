import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .causal_self_attention import KVCache
from .decoder_layers import SelfAttentionDecoderLayer
from .masks import CausalMask, PadMask
from .positional_encoding import PositionalEncoding


ModelCache = Tuple[KVCache, ...]

## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    '''
    A Pre-LN Decoder-Only Transformer model.
    '''

    INITIALIZATION_STD = 0.02
    def __init__(
            self, 
            num_layers: int, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
            num_kv_heads: Optional[int] = None,
            rope_base: float = 10_000.0,
            use_rope: bool = True,
    ) -> None:
        '''
        Initialize the Decoder-Only Transformer model.

        Args:
            num_layers: int, number of decoder layers
            d_model: int, model dimension
            num_heads: int, number of attention heads
            d_ff: int, feed-forward dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
            num_kv_heads: Optional[int], number of key/value heads (default: MHA)
            rope_base: float, frequency base used by rotary positions
            use_rope: bool, whether to use rotary positions (default: True)
        '''
        super().__init__()

        # DO NOT MODIFY: public configuration and cache contracts.
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        if not 0.0 <= layer_drop_rate < 1.0:
            raise ValueError("layer_drop_rate must be in the range [0, 1)")
        if not use_rope and num_kv_heads not in (None, num_heads):
            raise ValueError("num_kv_heads requires use_rope=True")

        # Initialize the decoder
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len         = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes     = num_classes
        self.num_layers      = num_layers
        self.d_model         = d_model
        self.num_heads       = num_heads
        self.num_kv_heads    = num_heads if num_kv_heads is None else num_kv_heads
        self.use_rope        = use_rope

        # ModuleList of decoder layers
        self.dec_layers = nn.ModuleList([
            SelfAttentionDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_rope=use_rope,
                max_len=max_len,
                num_kv_heads=self.num_kv_heads,
                rope_base=rope_base,
            )
            for _ in range(num_layers)
        ])

        # Token embedding and projection
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.final_linear = nn.Linear(d_model, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # DO NOT MODIFY
        self.apply(self._initialize_module)
        self._scale_residual_projections()

        # Weight tying (extra form of regularization, read more about it)
        if weight_tying:
            self.final_linear.weight = self.target_embedding.weight

    @classmethod
    def _initialize_module(cls, module: nn.Module) -> None:
        """Initialize decoder modules with the GPT convention."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=cls.INITIALIZATION_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=cls.INITIALIZATION_STD)
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(
                module.in_proj_weight,
                mean=0.0,
                std=cls.INITIALIZATION_STD,
            )
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    def _scale_residual_projections(self) -> None:
        """Scale decoder residual outputs by depth."""
        residual_std = self.INITIALIZATION_STD / math.sqrt(2 * self.num_layers)
        for layer in self.dec_layers:
            nn.init.normal_(
                layer.self_attn.mha.out_proj.weight,
                mean=0.0,
                std=residual_std,
            )
            nn.init.normal_(
                layer.ffn.ffn[3].weight,
                mean=0.0,
                std=residual_std,
            )

    def _past_length(
        self,
        past_key_values: Optional[ModelCache],
    ) -> int:
        '''Validate a model cache and return its shared sequence length.'''
        # DO NOT MODIFY: layer-local batch, KV-head, head-dimension, dtype,
        # and device checks happen inside CausalSelfAttention._validate_cache.
        if past_key_values is None:
            return 0
        if not isinstance(past_key_values, (tuple, list)):
            raise ValueError("past_key_values must contain one cache per layer")
        if len(past_key_values) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer caches, got {len(past_key_values)}"
            )

        lengths = set()
        for layer_index, layer_cache in enumerate(past_key_values):
            if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
                raise ValueError(
                    f"Cache for layer {layer_index} must be a (key, value) pair"
                )
            key, value = layer_cache
            if not torch.is_tensor(key) or not torch.is_tensor(value):
                raise ValueError("Cached keys and values must be tensors")
            if key.ndim != 4 or value.ndim != 4:
                raise ValueError("Cached keys and values must be rank-4 tensors")
            if key.shape != value.shape:
                raise ValueError("Cached keys and values must have identical shapes")
            lengths.add(key.size(2))

        if len(lengths) != 1:
            raise ValueError(
                "All decoder-layer caches must have the same sequence length"
            )
        return lengths.pop()

    def forward(self, padded_targets: torch.Tensor, target_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        '''
        Forward pass for the decoder. Used for Training only. Tokens are assumed to be right-padded.
        Args:
            padded_targets (torch.Tensor): The padded target sequence. shape: (batch_size, seq_len)
            target_lengths (Optional[torch.Tensor]): The lengths of the target sequences. shape: (batch_size,)
        Returns:
            seq_out (torch.Tensor): The output sequence. shape: (batch_size, seq_len, d_model)
            runnint_att (dict): The attention weights. shape: (batch_size, seq_len, seq_len)
        '''
        # DO NOT MODIFY: HW4P1 uses fixed, unpadded language-model chunks, so
        # target_lengths is optional during both training and evaluation.
        if padded_targets.ndim != 2:
            raise ValueError(
                "padded_targets must have shape (batch_size, sequence_length)"
            )
        if padded_targets.size(1) == 0:
            raise ValueError("padded_targets must contain at least one token")
        if padded_targets.size(1) > self.max_len:
            raise ValueError(
                f"Context length {padded_targets.size(1)} exceeds max_len={self.max_len}"
            )

        if target_lengths is not None:
            expected_batch = padded_targets.size(0)
            if target_lengths.ndim != 1 or target_lengths.size(0) != expected_batch:
                raise ValueError("target_lengths must have shape (batch_size,)")
            expected_lengths = torch.full_like(
                target_lengths,
                padded_targets.size(1),
            )
            if self.use_rope and not torch.equal(target_lengths, expected_lengths):
                raise ValueError(
                    "HW4P1 uses fixed, unpadded chunks; all lengths must equal T"
                )
        
        # Create padding mask if lengths are supplied
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets, target_lengths)
        
        # Causal mask for additive positional path; RoPE builds rectangular causal masks inside attention
        causal_mask = None
        if not self.use_rope:
            causal_mask = CausalMask(padded_targets).to(padded_targets.device)

        # Token embedding
        x = self.target_embedding(padded_targets)
        if not self.use_rope:
            x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through decoder layers
        runnint_att = {}
        for i in range(self.num_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            
            x, attention = self.dec_layers[i](
                x,
                key_padding_mask=pad_mask_dec,
                attn_mask=causal_mask,
            )
            runnint_att['layer{}_dec_self'.format(i + 1)] = attention

        # Final normalization and vocab projection
        x = self.norm(x)
        seq_out = self.final_linear(x)

        return seq_out, runnint_att
    
    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        '''
        Score the tokens for the decoder. 
        This is used for scoring the next token for a given prompt.
        Padding mask is not applied so ensure that the prompts are not padded. 
        Can only handle batch_size = 1 or batch with same lengths and no padding. 
        Args:
            prompts (torch.Tensor) : tensor of fixed length token sequences. shape: (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): Batch of next token logits. shape: (batch_size, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training, use forward method instead")
        # Forward pass with no target lengths
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        # Return the last token's logits for next token prediction    
        logits     = seq_out[:, -1, :]
        return logits

    def forward_cached(
        self,
        padded_targets: torch.Tensor,
        past_key_values: Optional[ModelCache] = None,
    ) -> Tuple[torch.Tensor, dict, ModelCache]:
        '''
        Process only new token IDs and progressively extend every
        decoder layer's compact KV cache.
        '''
        # DO NOT MODIFY: cached decoding is rotary inference only.
        if self.training:
            raise ValueError("KV caching is inference-only; call model.eval() first")
        if not self.use_rope:
            raise ValueError("KV caching requires use_rope=True")
        if padded_targets.ndim != 2:
            raise ValueError(
                "padded_targets must have shape (batch_size, sequence_length)"
            )
        if padded_targets.size(1) == 0:
            raise ValueError("padded_targets must contain at least one token")

        past_length = self._past_length(past_key_values)
        context_length = past_length + padded_targets.size(1)
        if context_length > self.max_len:
            raise ValueError(
                f"Context length {context_length} exceeds max_len={self.max_len}"
            )

        # Embed and dropout only the new tokens
        x = self.target_embedding(padded_targets)
        x = self.dropout(x)

        running_att = {}
        present_key_values = []
        layer_caches = (
            [None] * self.num_layers
            if past_key_values is None
            else past_key_values
        )
        for layer_index, (layer, layer_cache) in enumerate(
            zip(self.dec_layers, layer_caches)
        ):
            x, attention, present_kv = layer.forward_cached(
                x,
                past_kv=layer_cache,
            )
            running_att[f"layer{layer_index + 1}_dec_self"] = attention
            present_key_values.append(present_kv)

        # Final normalization and vocab projection
        x = self.norm(x)
        logits = self.final_linear(x)
        present = tuple(present_key_values)

        return logits, running_att, present

    def score_cached(
        self,
        padded_targets: torch.Tensor,
        past_key_values: Optional[ModelCache] = None,
    ) -> Tuple[torch.Tensor, ModelCache]:
        '''Return next-token logits and the extended model cache.'''
        logits, _, present_key_values = self.forward_cached(
            padded_targets,
            past_key_values=past_key_values,
        )
        next_token_logits = logits[:, -1, :]
        return next_token_logits, present_key_values