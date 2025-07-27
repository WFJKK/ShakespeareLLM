import torch
import torch.nn as nn
from .causalmask import causal_mask


class TokenEmbedding(nn.Module):
    """
    Embeds input token indices into dense vectors.
    
    Args:
        input_dim (int): Size of the vocabulary.
        model_dim (int): Dimensionality of the embedding vectors.
    """
    def __init__(self, input_dim, model_dim): 
        super().__init__()
        self.TE = nn.Embedding(input_dim, model_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor of token indices with shape (B, T)

        Returns:
            Tensor: Embedded token representations with shape (B, T, model_dim)
        """
        return self.TE(x)


class MultiheadwRotataryEmbedding(nn.Module): 
    """
    Multi-head self-attention layer with rotary positional embeddings.

    Args:
        model_dim (int): Dimensionality of the model.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """
    def __init__(self, model_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads 
        self.head_dim = model_dim // n_heads
        assert model_dim % n_heads == 0
        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_o = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, D)

        Returns:
            Tensor: Output tensor after multi-head attention of shape (B, T, D)
        """
        B, T, D = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        Q = self.rot(Q)
        K = self.rot(K)

        att_scores = Q @ K.transpose(-1, -2)
        mask = causal_mask(T, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, T, T)
        masked_att_scores = att_scores + mask
        score = torch.softmax(masked_att_scores / (self.head_dim ** 0.5), dim=-1)
        score = self.dropout(score)
        output = score @ V
        output = output.permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.W_o(output)

    @staticmethod
    def rot(x):
        """
        Applies rotary positional embedding.

        Args:
            x (Tensor): Tensor of shape (B, n_heads, T, head_dim)

        Returns:
            Tensor: Rotary positionally encoded tensor of same shape
        """
        _, _, T, D = x.shape
        device = x.device
        half_dim = D // 2
        assert D % 2 == 0, "D (head_dim) must be even"
        theta = 1 / (10000 ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
        pfactor = torch.arange(0, T, device=device)
        angle = torch.einsum('i,j->ij', pfactor, theta)
        sin = torch.sin(angle)[None, None, :, :]
        cos = torch.cos(angle)[None, None, :, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)


class FeedForward(nn.Module):
    """
    Feed-forward network block used within Transformer.

    Args:
        model_dim (int): Input/output dimensionality.
        hidden_dim (int): Hidden layer dimensionality.
        dropout (float): Dropout rate.
    """
    def __init__(self, model_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, model_dim)

        Returns:
            Tensor: Output tensor of same shape
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer block with pre-LayerNorm and residual connections.

    Args:
        model_dim (int): Model dimensionality.
        hidden_dim (int): Feedforward network hidden size.
        dropout (float): Dropout rate.
        n_heads (int): Number of attention heads.
    """
    def __init__(self, model_dim, hidden_dim, dropout, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mha = MultiheadwRotataryEmbedding(model_dim, n_heads, dropout)
        self.ffn = FeedForward(model_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, model_dim)

        Returns:
            Tensor: Output tensor of same shape
        """
        residual = x
        att = self.mha(self.norm1(x))
        y = att + residual
        ffn_out = self.ffn(self.norm2(y))
        return y + self.dropout(ffn_out)


class TransformerModel(nn.Module):
    """
    Full Transformer model for language modeling.

    Args:
        input_dim (int): Vocabulary size.
        model_dim (int): Embedding and hidden dimensionality.
        hidden_dim (int): Feedforward network hidden size.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of Transformer blocks.
        dropout (float): Dropout rate.
    """
    def __init__(self, input_dim, model_dim, hidden_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.token_embedding = TokenEmbedding(input_dim, model_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(model_dim, hidden_dim, dropout, n_heads)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T) containing token indices

        Returns:
            Tensor: Logits of shape (B, T, input_dim)
        """
        x = self.token_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


