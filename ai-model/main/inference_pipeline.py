"""Inference pipeline for the Two-Tower recommendation model.

Provides a RecommendationPipeline class that loads all required models
and data once, then serves recommendations efficiently.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from pymilvus import Collection

from util.util import (
    connect_to_milvus,
    get_user_history,
    get_user_vector,
    load_encoders,
    load_user_tower,
    query_milvus,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """End-to-end recommendation pipeline.

    Loads all required models, encoders, and data once during
    initialization, then serves recommendations via `recommend()`.

    Args:
        model_dir: Directory containing trained model checkpoints.
        data_dir: Directory containing processed data and encoders.
        collection_name: Name of the Milvus collection to query.
    """

    def __init__(
        self,
        model_dir: str = "model",
        data_dir: str = "processed_data",
        collection_name: str = "product_vectors",
    ) -> None:
        logger.info("Initializing recommendation pipeline...")

        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

        # Load encoders
        self.product_encoder, self.brand_encoder = load_encoders(str(self.data_dir))

        # Load user features
        self.user_features_df = pd.read_parquet(self.data_dir / "user_features.parquet")

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load UserTower
        checkpoint_path = self.model_dir / "user_tower_checkpoint.pth"
        self.user_tower = load_user_tower(
            str(checkpoint_path),
            product_vocab_size=len(self.product_encoder.classes_),
            device=self.device,
        )

        # Connect to Milvus
        if not connect_to_milvus():
            raise RuntimeError("Cannot connect to Milvus.")

        self.collection = Collection(name=collection_name)
        self.collection.load()

        logger.info("Pipeline ready.")

    def recommend(self, user_id: str, k: int = 10) -> list[dict]:
        """Generate top-k recommendations for a given user.

        Args:
            user_id: The user's reviewer ID.
            k: Number of recommendations to return.

        Returns:
            List of recommendation dicts with product_id, score, and rank.
            Empty list if user not found or history is too short.
        """
        history = get_user_history(user_id, self.user_features_df)
        if history is None:
            return []

        user_vector = get_user_vector(history, self.user_tower, self.device)
        return query_milvus(user_vector, self.collection, k=k)


def main() -> None:
    """Run a test recommendation for the first user in the dataset."""
    pipeline = RecommendationPipeline()

    sample_user = pipeline.user_features_df.iloc[0]["reviewerID"]

    print(f"\n=== TEST RECOMMENDATION ===")
    print(f"User: {sample_user}")

    recommendations = pipeline.recommend(sample_user, k=10)

    if not recommendations:
        print("No recommendations found.")
        return

    for rec in recommendations:
        print(f"{rec['rank']:2d}. {rec['product_id']}  score: {rec['score']:.4f}")


if __name__ == "__main__":
    main()
