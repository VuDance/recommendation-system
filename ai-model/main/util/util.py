"""Utility functions for the recommendation system."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from pymilvus import connections

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

