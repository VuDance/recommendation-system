"""Compute product vectors using the trained ProductTower model.

This module provides functions to load a trained ProductTower model
and generate embedding vectors for all products in the catalog.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.product_tower import ProductTower
from main.util.util import load_encoders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_product_tower(checkpoint_path: str, brand_vocab_size: int) -> ProductTower:
    """Load a trained ProductTower model from a checkpoint.

    Args:
        checkpoint_path: Path to the saved model weights.
        brand_vocab_size: Number of unique brands in the vocabulary.

    Returns:
        ProductTower model in evaluation mode.
    """
    logger.info("Loading ProductTower from %s", checkpoint_path)
    product_tower = ProductTower(brand_vocab_size=brand_vocab_size)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    product_tower.load_state_dict(checkpoint["model_state_dict"])
    product_tower.eval()

    logger.info("ProductTower loaded successfully.")
    return product_tower


def compute_product_vectors(
    data_dir: str = "processed_data",
    model_checkpoint: str = "model/product_tower_checkpoint.pth",
    batch_size: int = 128,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Generate product embedding vectors for the entire catalog.

    Args:
        data_dir: Directory containing processed data and encoders.
        model_checkpoint: Path to the trained ProductTower checkpoint.
        batch_size: Items to process per inference batch.

    Returns:
        Tuple of (vectors, product_ids, product_idxs).
    """
    data_path = Path(data_dir)

    # Load data
    logger.info("Loading item features...")
    item_df = pd.read_parquet(data_path / "item_features.parquet")
    item_df = item_df.reset_index(drop=True)
    logger.info("Loaded %d products.", len(item_df))

    # Load encoders and model
    _, brand_encoder = load_encoders(str(data_path))
    product_tower = load_product_tower(model_checkpoint, brand_vocab_size=len(brand_encoder.classes_))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    product_tower = product_tower.to(device)

    # Load SentenceTransformer for text encoding
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_vectors: list[np.ndarray] = []
    all_product_ids: list[str] = []
    all_product_idxs: list[int] = []

    logger.info("Computing product vectors...")

    for i in range(0, len(item_df), batch_size):
        batch_df = item_df.iloc[i : i + batch_size]
        text_inputs = [
            f"{row['title']} {row['description']}"
            for _, row in batch_df.iterrows()
        ]

        # Compute text embeddings
        text_embeddings = text_model.encode(
            text_inputs, convert_to_tensor=True, show_progress_bar=False
        )

        brand_ids = torch.tensor(batch_df["brand_id"].values, dtype=torch.long)
        prices = torch.zeros(len(batch_df), dtype=torch.float32)

        with torch.no_grad():
            product_vectors = product_tower(
                text_embeddings.to(device),
                brand_ids.to(device),
                prices.to(device),
            )

        all_vectors.append(product_vectors.cpu().numpy())
        all_product_ids.extend(batch_df["product_id"].tolist())
        all_product_idxs.extend(range(i, i + len(batch_df)))

        if (i // batch_size + 1) % 10 == 0:
            logger.info("Processed %d/%d products", i + len(batch_df), len(item_df))

    all_vectors_np = np.vstack(all_vectors)
    logger.info("Final vectors shape: %s", all_vectors_np.shape)

    # Save vectors
    output_dir = data_path
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "product_vectors.npy", all_vectors_np)

    metadata: dict[str, Any] = {
        "product_ids": all_product_ids,
        "product_idxs": all_product_idxs,
        "vectors_shape": all_vectors_np.shape,
    }
    with open(output_dir / "product_vectors_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    logger.info("Product vectors saved successfully!")
    return all_vectors_np, all_product_ids, all_product_idxs


def main() -> None:
    """Entry point for product vector computation."""
    try:
        vectors, product_ids, product_idxs = compute_product_vectors()

        print("\n=== PRODUCT VECTOR COMPUTATION COMPLETE ===")
        print(f"Vectors shape: {vectors.shape}")
        print(f"Sample vector (first 5 dims): {vectors[0][:5]}")
        print(f"Product IDs sample: {product_ids[:5]}")
        print(f"Product indices sample: {product_idxs[:5]}")
    except Exception as exc:
        logger.error("Error computing product vectors: %s", exc)
        raise


if __name__ == "__main__":
    main()
