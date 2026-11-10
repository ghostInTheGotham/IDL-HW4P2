import numpy as np
from typing import Optional

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """
    def __init__(self) -> None:
        self.Q = None
        self.K = None
        self.V = None
        self.mask = None
        self.attention_scores = None

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        :param Q: (..., L, d_k)
        :param K: (..., S, d_k)
        :param V: (..., S, d_v)
        :param mask: (..., L, S) where True/1 indicates positions to ignore
        :return: (..., L, d_v)
        """
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

        d_k = Q.shape[-1]

        # 1. Scaled dot product: (..., L, S)
        scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)

        # 2. Check for fully masked rows across the key dimension S
        all_masked_rows = None
        if mask is not None:
            mask_bool = mask.astype(bool)
            scores = np.where(mask_bool, -1e9, scores)
            all_masked_rows = np.all(mask_bool, axis=-1, keepdims=True)

        # 3. Stable Softmax
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)

        if mask is not None:
            exp_scores = np.where(mask.astype(bool), 0.0, exp_scores)

        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
        # Prevent division by zero for rows that are completely masked
        sum_exp_safe = np.where(sum_exp == 0, 1.0, sum_exp)

        self.attention_scores = exp_scores / sum_exp_safe

        # Enforce zero probability on completely masked rows
        if all_masked_rows is not None:
            self.attention_scores = np.where(all_masked_rows, 0.0, self.attention_scores)

        # 4. Context vectors: (..., L, d_v)
        output = np.matmul(self.attention_scores, V)
        return output

    def backward(
        self,
        d_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param d_output: (..., L, d_v)
        :return: (d_Q, d_K, d_V)
        """
        d_k = self.Q.shape[-1]

        # Gradient w.r.t. V: (..., S, d_v)
        d_V = np.matmul(np.swapaxes(self.attention_scores, -2, -1), d_output)

        # Gradient w.r.t. attention_scores: (..., L, S)
        d_scores_attn = np.matmul(d_output, np.swapaxes(self.V, -2, -1))

        # Backpropagation through Softmax
        # d_logits = P * (d_P - sum(d_P * P, axis=-1, keepdims=True))
        sum_d_scores_attn_p = np.sum(d_scores_attn * self.attention_scores, axis=-1, keepdims=True)
        d_scores = self.attention_scores * (d_scores_attn - sum_d_scores_attn_p)

        # Zero out gradients for masked positions
        if self.mask is not None:
            d_scores = np.where(self.mask.astype(bool), 0.0, d_scores)

        # Scale by 1 / sqrt(d_k)
        d_scores = d_scores / np.sqrt(d_k)

        # Gradients w.r.t Q and K
        d_Q = np.matmul(d_scores, self.K)
        d_K = np.matmul(np.swapaxes(d_scores, -2, -1), self.Q)

        return d_Q, d_K, d_V