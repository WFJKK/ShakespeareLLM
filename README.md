# Shakespeare LLM

A (modular)  Transformer trained on the standard `shakespeare.txt`. This project implements modern decoder-only Transformer using PyTorch and supports training with masked causal attention, and name sampling via autoregression. Includes modern architecture elements such as pre layernorm, feedforward, dropout, residual connections.
Also includes rotary positional embeddings. Uses GPT2 BPE tokenizer.

---

##  Project Structure

- `/shakespearepackage`: main package files
- `actual_generation.py`: script for text generation
- `traing.py`:training file
- `requirements.txt`: dependencies
- `/data`: includes shakespeare.txt, which is the text file used for training and validation



---

##  Features

- Transformer-style decoder-only architecture
- Rotary positional embeddings
- Causal masking 
- Modular, readable PyTorch implementation
- Autoregressive text generation
- W&B integration for logging
- Checkpointing and resume support

---

##  Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/WFJKK/ShakespeareLLM
   cd ShakespeareLLM
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Training

To train the model on `shakespeare.txt` w/:

 training.py


- Automatically uses GPU (MPS or CUDA) if available
- Logs training loss to [Weights & Biases](https://wandb.ai/)
- Automatically checkpoints to `checkpoint.pth` every 10 epochs
- Resumes from checkpoint if one exists

Modify `config.py` to adjust hyperparameters such as :
- Model dimensions
- Number of layers/heads
- Learning rate, batch size, etc.

---

##  Generate text

After training you can sample Shakespeare-style text:

 actual_generation.py


---



## License

This project is licensed under the **MIT License** — feel free to use, modify, or contribute.

---

