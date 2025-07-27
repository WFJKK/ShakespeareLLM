"""
Text generation utilities for ShakespeareLLM, including unconditional
autoregressive sampling with temperature control.
"""

import torch
from .config import *

def generate_unconditional(model, tokenizer, max_new_tokens=100, temperature=1.0, device=None):
    """
    Generate text unconditionally by sampling tokens sequentially from the model's output distribution.

    Args:
        model (nn.Module): Trained Transformer model.
        tokenizer (PreTrainedTokenizer): Tokenizer used to encode/decode text.
        max_new_tokens (int): Number of tokens to generate.
        temperature (float): Sampling temperature; higher values increase randomness.
        device (torch.device or None): Device on which to run generation.

    Returns:
        str: Generated text string.
    """
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 50256
    input_ids = torch.tensor([[start_token_id]], device=device)

    for _ in range(max_new_tokens):
        input_crop = input_ids[:, -block_size:]
        with torch.no_grad():
            logits = model(input_crop)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    return tokenizer.decode(input_ids[0])
