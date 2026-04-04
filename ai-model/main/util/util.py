"""Utility functions for the recommendation system."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from pymilvus import connections

from main.user_tower import UserTower

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


def load_user_tower(checkpoint_path: str, product_vocab_size: int,
                    device: str = "cpu") -> UserTower:
    """Load a trained UserTower model from a checkpoint.

    Args:
        checkpoint_path: Path to the saved model checkpoint.
        product_vocab_size: Size of the product vocabulary.
        device: Device to load the model onto.

    Returns:
        UserTower model in evaluation mode.
    """
    logger.info("Loading UserTower from %s", checkpoint_path)
    user_tower = UserTower(product_vocab_size=product_vocab_size).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    user_tower.load_state_dict(checkpoint["model_state_dict"])
    user_tower.eval()

    return user_tower


def get_user_history(user_id: str, user_features_df: Any) -> list[int] | None:
    """Retrieve and clean a user's purchase history.

    Args:
        user_id: The user's reviewer ID.
        user_features_df: DataFrame containing user features.

    Returns:
        List of product indices (excluding padding), or None if user not found.
    """
    user_row = user_features_df[user_features_df["reviewerID"] == user_id]
    if user_row.empty:
        logger.warning("User %s not found.", user_id)
        return None

    history = user_row["history_padded"].iloc[0]
    actual_history = [int(p) for p in history if p != 0]

    if len(actual_history) < 3:
        logger.warning("User %s history too short: %d items.", user_id, len(actual_history))
        return None

    return actual_history


def get_user_vector(user_history: list[int], user_tower: UserTower,
                    device: torch.device) -> Any:
    """Compute a user embedding vector from their purchase history.

    Args:
        user_history: List of product indices from the user's history.
        user_tower: Trained UserTower model.
        device: Device to run inference on.

    Returns:
        NumPy array of shape (output_dim,).
    """
    history_tensor = torch.tensor(user_history, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        user_vector = user_tower(history_tensor.to(device))
    return user_vector.cpu().numpy().flatten()


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

