"""Product Tower module for Two-Tower recommendation model.

Encodes product features (text embeddings, brand ID, price)
into a fixed-dimensional product vector.
"""

import torch


class ProductTower(torch.nn.Module):
    """Encodes product features into a normalized embedding vector.

    Args:
        brand_vocab_size: Number of unique brands in the vocabulary.
        text_dim: Dimensionality of input text embeddings (default: 384 for MiniLM).
        brand_embed_dim: Dimensionality of brand embedding (default: 16).
        price_embed_dim: Dimensionality of price embedding (default: 16).
        hidden_dim: Hidden layer size in the combined encoder (default: 128).
        output_dim: Final output embedding dimensionality (default: 64).
    """

    def __init__(
        self,
        brand_vocab_size: int,
        text_dim: int = 384,
        brand_embed_dim: int = 16,
        price_embed_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ) -> None:
        super().__init__()

        self.brand_embedding = torch.nn.Embedding(brand_vocab_size, brand_embed_dim)
        self.text_encoder = torch.nn.Linear(text_dim, 64)
        self.price_encoder = torch.nn.Linear(1, price_embed_dim)

        combined_input_dim = 64 + brand_embed_dim + price_embed_dim

        self.combined_encoder = torch.nn.Sequential(
            torch.nn.Linear(combined_input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        brand_ids: torch.Tensor,
        prices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Product Tower.

        Args:
            text_features: Text embeddings of shape (batch, text_dim).
            brand_ids: Brand indices of shape (batch,).
            prices: Price values of shape (batch,).

        Returns:
            L2-normalized product embeddings of shape (batch, output_dim).
        """
        text_encoded = self.text_encoder(text_features)
        brand_embedded = self.brand_embedding(brand_ids)
        price_encoded = self.price_encoder(prices.unsqueeze(-1))

        combined = torch.cat([text_encoded, brand_embedded, price_encoded], dim=-1)
        output = self.combined_encoder(combined)

        return torch.nn.functional.normalize(output, p=2, dim=-1)
