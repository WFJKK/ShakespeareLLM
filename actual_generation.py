"""
Script to load a trained ShakespeareLLM Transformer model, initialize
the GPT-2 tokenizer, and generate text unconditionally using the
provided generation function. Automatically selects device (MPS/CUDA/CPU).
"""


import torch
from transformers import GPT2Tokenizer
from shakespearepackage.model import TransformerModel
from shakespearepackage.config import *
from shakespearepackage.generation_function import generate_unconditional

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA (GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")



model = TransformerModel(
    input_dim=tokenizer.vocab_size,
    model_dim=model_dim,
    hidden_dim=hidden_dim,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=0.1  
).to(device)

checkpoint_path = "best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()




print(generate_unconditional(model, tokenizer, max_new_tokens=150, temperature=1.0,device =device ))

