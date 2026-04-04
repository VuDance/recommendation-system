"""User Tower module for Two-Tower recommendation model.

Encodes user purchase history into a fixed-dimensional user vector.
"""

import torch


class UserTower(torch.nn.Module):
    """Encodes user history into a normalized embedding vector.

    Args:
        product_vocab_size: Number of unique products in the vocabulary.
        product_embed_dim: Dimensionality of product embedding (default: 32).
        hidden_dim: GRU hidden state size (default: 128).
        num_layers: Number of GRU layers (default: 2).
        output_dim: Final output embedding dimensionality (default: 64).
    """

    def __init__(
        self,
        product_vocab_size: int,
        product_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 64,
    ) -> None:
        super().__init__()

        self.product_embedding = torch.nn.Embedding(product_vocab_size, product_embed_dim)
        self.history_encoder = torch.nn.GRU(
            input_size=product_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.output_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.LayerNorm(output_dim),
        )

    def forward(self, product_history: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the User Tower.

        Args:
            product_history: Padded product IDs of shape (batch, seq_len).
            lengths: Actual sequence lengths of shape (batch,), used for packing.
                     If None, uses the last hidden state directly.

        Returns:
            L2-normalized user embeddings of shape (batch, output_dim).
        """
        embedded = self.product_embedding(product_history)

        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.history_encoder(packed)
        else:
            _, hidden = self.history_encoder(embedded)

        output = self.output_encoder(hidden[-1])
        return torch.nn.functional.normalize(output, p=2, dim=-1)
