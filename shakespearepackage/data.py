"""
Loads Shakespeare text data, tokenizes with GPT-2 tokenizer, prepares
train/validation splits, and creates DataLoader instances for training.
Suppresses unnecessary warnings and Hugging Face logging output.
"""
import os 
import torch
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from .config import *
from .Dataset import ShakespeareDataset
import warnings
from transformers.utils import logging as hf_logging



hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
base_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(base_dir, '..', 'data', 'shakespeare.txt')
filepath = os.path.abspath(filepath) 

with open(filepath, 'r') as file:
    text = file.read()



vocab_size = tokenizer.vocab_size
encodings = tokenizer(text)  
input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)

train_ratio = 0.9
train_idx = int(train_ratio * len(input_ids))
train_ids = input_ids[:train_idx]
val_ids = input_ids[train_idx:]

train_data = ShakespeareDataset(train_ids, block_size=block_size)
val_data = ShakespeareDataset(val_ids, block_size=block_size)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
