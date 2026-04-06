"""Utility functions for the recommendation system."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pymilvus import connections
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv(Path(__file__).parents[2] / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_encoders(data_dir: str = "processed_data"):
    """Load fitted label encoders from disk.

    Args:
        data_dir: Directory containing the encoder pickle files.

    Returns:
        Tuple of (product_encoder, brand_encoder).
    """
    logger.info("Loading encoders...")
    data_path = Path(data_dir)

    with open(data_path / "product_encoder.pkl", "rb") as f:
        product_encoder = pickle.load(f)

    with open(data_path / "brand_encoder.pkl", "rb") as f:
        brand_encoder = pickle.load(f)

    return product_encoder, brand_encoder


def connect_to_milvus(host: str | None = None, port: str | None = None,
                     user: str | None = None, password: str | None = None) -> bool:
    """Establish connection to Milvus vector database.

    Args:
        host: Milvus server hostname (falls back to MILVUS_HOST env var).
        port: Milvus server port (falls back to MILVUS_PORT env var).
        user: Milvus username (falls back to MILVUS_USER env var).
        password: Milvus password (falls back to MILVUS_PASSWORD env var).

    Returns:
        True if connection succeeds, False otherwise.
    """
    try:
        connections.connect(
            alias="default",
            host=host or os.getenv("MILVUS_HOST", "localhost"),
            port=port or os.getenv("MILVUS_PORT", "19530"),
            user=user or os.getenv("MILVUS_USER", "root"),
            password=password or os.getenv("MILVUS_PASSWORD", "Milvus"),
        )
        logger.info("Connected to Milvus.")
        return True
    except Exception as exc:
        logger.error(f"Milvus connection failed: {exc}")
        return False


def query_milvus(user_vector: Any, collection: Any, k: int = 10) -> list[dict]:
    """Search Milvus for the top-k most similar product vectors.

    Args:
        user_vector: User embedding array of shape (dim,).
        collection: Loaded Milvus collection to search.
        k: Number of recommendations to return.

    Returns:
        List of dicts with keys: product_id, score, rank.
    """
    results = collection.search(
        data=[user_vector.tolist()],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=k,
        output_fields=["product_id"],
    )

    recommendations = []
    for hits in results:
        for rank, hit in enumerate(hits, 1):
            recommendations.append({
                "product_id": hit.entity.get("product_id"),
                "score": hit.distance,
                "rank": rank,
            })

    return recommendations


def get_user_history(user_id: str, user_features_df: pd.DataFrame) -> pd.Series | None:
    """Get a user's purchase history from the user features DataFrame.

    Args:
        user_id: The user's reviewer ID.
        user_features_df: DataFrame with columns reviewerID and history_padded.

    Returns:
        Series with user's history or None if user not found.
    """
    user_row = user_features_df[user_features_df["reviewerID"] == user_id]
    if user_row.empty:
        logger.warning("User %s not found in user features.", user_id)
        return None

    history = user_row.iloc[0]["history_padded"]
    # Remove padding zeros
    history_list = [int(p) for p in history if p != 0]
    if len(history_list) < 2:
        logger.warning("User %s has insufficient history (%d items).", user_id, len(history_list))
        return None

    return user_row.iloc[0]


def get_user_vector(
    history: pd.Series,
    user_tower: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Compute user embedding from their interaction history.

    Args:
        history: User history row from user_features DataFrame.
        user_tower: Loaded UserTower model.
        device: Torch device for inference.

    Returns:
        User embedding vector of shape (output_dim,).
    """
    history_padded = history["history_padded"]
    history_tensor = torch.tensor([history_padded], dtype=torch.long)
    length = torch.tensor([sum(1 for x in history_padded if x != 0)], dtype=torch.long)

    with torch.no_grad():
        user_vec = user_tower(history_tensor.to(device), lengths=length)
    return user_vec.cpu().numpy().flatten()


def load_user_tower(
    checkpoint_path: str,
    product_vocab_size: int,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a trained UserTower model from a checkpoint.

    Args:
        checkpoint_path: Path to the saved model weights.
        product_vocab_size: Number of unique products in the vocabulary.
        device: Torch device for the model.

    Returns:
        UserTower model in evaluation mode.
    """
    from main.user_tower import UserTower

    logger.info("Loading UserTower from %s", checkpoint_path)
    user_tower = UserTower(
        product_vocab_size=product_vocab_size,
        output_dim=64,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    user_tower.load_state_dict(checkpoint["model_state_dict"])
    user_tower.eval()

    if device is not None:
        user_tower = user_tower.to(device)

    logger.info("UserTower loaded successfully.")
    return user_tower


def encode_texts_for_content(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode text into a dense embedding vector using SentenceTransformer.

    Args:
        text: The input text to encode.
        model_name: Name of the SentenceTransformer model to use.

    Returns:
        Numpy array of shape (embedding_dim,).
    """
    model = SentenceTransformer(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def get_content_based_recommendations(
    query_text: str,
    collection: Any,
    k: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """Get content-based recommendations by encoding query text and searching Milvus.

    Args:
        query_text: Text to search for (e.g., product description).
        collection: Loaded Milvus collection to search.
        k: Number of recommendations to return.
        model_name: Name of the SentenceTransformer model for encoding.

    Returns:
        List of dicts with keys: product_id, score, rank.
    """
    model = SentenceTransformer(model_name)
    query_vector = model.encode(query_text, convert_to_numpy=True).tolist()

    recommendations = query_milvus(query_vector, collection, k=k)
    return recommendations


