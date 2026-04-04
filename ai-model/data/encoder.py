"""Text encoding module using SentenceTransformer.

Provides a singleton pattern for loading the embedding model
and encoding metadata text into dense vectors.
"""

from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

_model: Optional[SentenceTransformer] = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Get or lazily initialize the SentenceTransformer model.

    Args:
        model_name: Name of the SentenceTransformer model to load.

    Returns:
        Loaded SentenceTransformer instance.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name, device=_device)
    return _model


def encode_metadata(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
    show_progress: bool = True,
) -> list[list[float]]:
    """Encode a list of text strings into embedding vectors.

    Args:
        texts: List of text strings to encode.
        model_name: Name of the SentenceTransformer model to use.
        batch_size: Number of texts to encode per batch.
        show_progress: Whether to display a progress bar.

    Returns:
        List of embedding vectors.
    """
    model = get_model(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress)
