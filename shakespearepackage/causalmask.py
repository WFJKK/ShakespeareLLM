"""
Causal masking utility for autoregressive Transformer models.
"""

import torch

def causal_mask(seq_len, device=None):
    """
    Creates a causal mask for sequence modeling.

    Args:
        seq_len (int): The length of the sequence to mask.
        

    Returns:
        A (seq_len, seq_len) tensor with -inf above the
        main diagonal and 0 elsewhere.
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask

