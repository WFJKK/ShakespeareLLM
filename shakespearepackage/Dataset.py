"""
Defines a dataset. Provides input-target pairs
where targets are inputs shifted by one token.
"""


from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    """
    A token-level dataset for training autoregressive language models.

    Args:
        input_ids (torch.Tensor): 1D tensor of token IDs.
        block_size (int): Length of each input sequence.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A pair (x, y) where:
            - x is a tensor of shape (block_size,) containing input tokens
            - y is a tensor of shape (block_size,) containing target tokens (input shifted by one)
    """

    def __init__(self, input_ids, block_size):  
        super().__init__()
        self.data = input_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


