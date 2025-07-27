"""
Loads training and validation data, initializes model, optimizer, and
loss criterion. Supports resuming from checkpoints, logs metrics to Weights & Biases,
and saves the best performing model based on validation loss.
Automatically selects device (MPS/CUDA/CPU).
"""




import torch
import torch.optim as optim 
import torch.nn as nn
import os
import wandb
from shakespearepackage.config import *
from shakespearepackage.model import TransformerModel
from shakespearepackage.data import train_loader, val_loader, vocab_size

wandb.init(project="shakespeare-transformer", config={
    "model_dim": model_dim,
    "hidden_dim": hidden_dim,
    "n_heads": n_heads,
    "n_layers": n_layers,
    "dropout": 0.1,
    "block_size": block_size,
    "batch_size": batch_size,
    "learning_rate": LR,
    "num_epochs": num_epochs
})


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA (GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")


model = TransformerModel(
    input_dim=vocab_size,
    model_dim=model_dim,
    hidden_dim=hidden_dim,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


checkpoint_path = "best_model.pt"
start_epoch = 0
best_val_loss = float("inf")

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed from epoch {start_epoch}, best val loss = {best_val_loss:.4f}")


print("Starting training loop")
for epoch in range(start_epoch, num_epochs):
    print(f"\nEpoch {epoch + 1}")
    model.train()
    total_train_loss = 0

    for input_tokens, target_tokens in train_loader:
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)

        logits = model(input_tokens)
        B, T, D = logits.shape
        loss = criterion(logits.view(B * T, D), target_tokens.view(B * T))
        
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for input_tokens, target_tokens in val_loader:
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            logits = model(input_tokens)
            B, T, D = logits.shape
            val_loss = criterion(logits.view(B * T, D), target_tokens.view(B * T))
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Val Loss: {avg_val_loss:.4f}")

    
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "best_val_loss": best_val_loss
    })

    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print("New best model saved.")

print("\nTraining loop ended")


