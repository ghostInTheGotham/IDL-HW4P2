from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np
from typing import Optional

class GroupedQueryAttention:
    """
    Grouped Query Attention
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> None:
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of query heads
        :param num_kv_heads: Number of key/value heads shared across query groups
        """
        # DO NOT MODIFY
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim     = embed_dim
        self.num_heads     = num_heads
        self.num_kv_heads  = num_kv_heads
        self.head_dim      = embed_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention()

        # Projections
        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj   = Linear(embed_dim, num_kv_heads * self.head_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(
        self,
        Wq: np.ndarray,
        bq: np.ndarray,
        Wk: np.ndarray,
        bk: np.ndarray,
        Wv: np.ndarray,
        bv: np.ndarray,
        Wo: np.ndarray,
        bo: np.ndarray,
    ) -> None:
        """
        Initialize the weights and biases with the given values.
        Note that Wk and Wv are shaped for the compact key/value heads.
        """
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        key_padding_mask: Optional[np.ndarray] = None,
        attn_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        # 1. Project inputs
        q = self.q_proj.forward(query)  # (N, L, num_heads * head_dim)
        k = self.k_proj.forward(key)    # (N, S, num_kv_heads * head_dim)
        v = self.v_proj.forward(value)  # (N, S, num_kv_heads * head_dim)

        # 2. Reshape for multiple heads
        q = self._split_heads(q, self.num_heads)        # (N, num_heads, L, head_dim)
        k = self._split_heads(k, self.num_kv_heads)     # (N, num_kv_heads, S, head_dim)
        v = self._split_heads(v, self.num_kv_heads)     # (N, num_kv_heads, S, head_dim)

        # 3. Repeat key/value heads for their query groups
        k = self._repeat_kv(k)  # (N, num_heads, S, head_dim)
        v = self._repeat_kv(v)  # (N, num_heads, S, head_dim)

        # 4. Combine padding and causal masks
        mask = self._merge_masks(key_padding_mask, attn_mask)

        # 5. Apply attention
        attn_outputs = self.attention.forward(q, k, v, mask)  # (N, num_heads, L, head_dim)

        # 6. Merge heads
        attn_output = self._concat_heads(attn_outputs)  # (N, L, embed_dim)

        # 7. Final projection
        output = self.out_proj.forward(attn_output)  # (N, L, embed_dim)

        return output

    def backward(
        self,
        d_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for grouped query attention.
        :param d_output: (N, L, E)
        :return: (d_query, d_key, d_value)
        """
        # 1. Backpropagate through output projection
        d_attn_output = self.out_proj.backward(d_output)  # (N, L, embed_dim)

        # 2. Split output gradient into heads
        d_attn_outputs = self._split_heads(d_attn_output, self.num_heads)  # (N, num_heads, L, head_dim)

        # 3. Backpropagate through scaled dot-product attention
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)

        # 4. Sum key and value gradients across groups
        d_k = self._sum_kv_gradients(d_k)  # (N, num_kv_heads, S, head_dim)
        d_v = self._sum_kv_gradients(d_v)  # (N, num_kv_heads, S, head_dim)

        # 5. Merge head gradients
        d_q = self._concat_heads(d_q)  # (N, L, num_heads * head_dim)
        d_k = self._concat_heads(d_k)  # (N, S, num_kv_heads * head_dim)
        d_v = self._concat_heads(d_v)  # (N, S, num_kv_heads * head_dim)

        # 6. Backpropagate through input projections
        d_query = self.q_proj.backward(d_q)
        d_key   = self.k_proj.backward(d_k)
        d_value = self.v_proj.backward(d_v)

        return d_query, d_key, d_value

    def _merge_masks(
        self,
        key_padding_mask: Optional[np.ndarray],
        attn_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Merge two mask types into a single mask.
        The masks are applied after the key/value heads have been repeated.
        """
        if key_padding_mask is None and attn_mask is None:
            return None

        key_mask = None
        if key_padding_mask is not None:
            # (N, S) -> (N, 1, 1, S)
            key_mask = key_padding_mask[:, np.newaxis, np.newaxis, :].astype(bool)

        attention_mask = None
        if attn_mask is not None:
            # (L, S) -> (1, 1, L, S)
            attention_mask = attn_mask[np.newaxis, np.newaxis, :, :].astype(bool)

        if key_mask is not None and attention_mask is not None:
            return key_mask | attention_mask
        elif key_mask is not None:
            return key_mask
        else:
            return attention_mask

    def _split_heads(
        self,
        x: np.ndarray,
        num_heads: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reshape tensor for multi-head attention.
        :param x: (N, T, num_heads * head_dim)
        :return: (N, num_heads, T, head_dim)
        """
        num_heads = self.num_heads if num_heads is None else num_heads
        N, T, _ = x.shape
        x = x.reshape(N, T, num_heads, self.head_dim)
        x = x.transpose(0, 2, 1, 3)
        return x

    def _concat_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Concatenate the heads back into a 3D tensor.
        :param x: (N, heads, T, head_dim)
        :return: (N, T, heads * head_dim)
        """
        N, heads, T, head_dim = x.shape
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(N, T, heads * head_dim)
        return x

    def _repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """
        Repeat each key/value head for the query heads that share it.
        :param x: (N, num_kv_heads, S, head_dim)
        :return: (N, num_heads, S, head_dim)
        """
        return np.repeat(x, self.num_kv_groups, axis=1)

    def _sum_kv_gradients(self, x: np.ndarray) -> np.ndarray:
        """
        Reduce repeated key/value gradients back to compact heads.
        :param x: (N, num_heads, S, head_dim)
        :return: (N, num_kv_heads, S, head_dim)
        """
        N, _, S, head_dim = x.shape
        x = x.reshape(N, self.num_kv_heads, self.num_kv_groups, S, head_dim)
        return x.sum(axis=2)