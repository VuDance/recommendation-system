"""Professional evaluation module for the Two-Tower recommendation model.

Implements standard recommendation system metrics:
- Recall@K, NDCG@K, Hit Rate@K, MRR@K
- Item coverage and popularity bias
- Per-user and aggregate statistics
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.product_tower import ProductTower
from main.user_tower import UserTower

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Evaluation Metrics ───────────────────────────────────


def recall_at_k(ranked_items: list[int], ground_truth: list[int], k: int) -> float:
    """Fraction of relevant items that appear in top-K recommendations."""
    if not ground_truth:
        return 0.0
    relevant_in_topk = len(set(ranked_items[:k]) & set(ground_truth))
    return relevant_in_topk / len(ground_truth)


def ndcg_at_k(ranked_items: list[int], ground_truth: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not ground_truth:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because position is 1-indexed

    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(ranked_items: list[int], ground_truth: list[int], k: int) -> float:
    """Whether at least one relevant item appears in top-K."""
    return 1.0 if set(ranked_items[:k]) & set(ground_truth) else 0.0


def mrr_at_k(ranked_items: list[int], ground_truth: list[int], k: int) -> float:
    """Mean Reciprocal Rank at K."""
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


# ── Evaluation Pipeline ──────────────────────────────────


class ModelEvaluator:
    """Evaluates a Two-Tower model on held-out test data.

    Args:
        data_dir: Path to processed data directory.
        model_dir: Path to model checkpoints.
        device: Torch device for inference.
    """

    def __init__(
        self,
        data_dir: str = "data/processed_data",
        model_dir: str = "model",
        device: str | None = None,
    ) -> None:
        self.data_path = Path(data_dir)
        self.model_path = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load data
        self._load_data()
        self._load_models()

    def _load_data(self) -> None:
        """Load test data and encoders."""
        logger.info("Loading data...")

        with open(self.data_path / "product_encoder.pkl", "rb") as f:
            self.product_encoder = pickle.load(f)
        with open(self.data_path / "brand_encoder.pkl", "rb") as f:
            self.brand_encoder = pickle.load(f)

        self.item_features = pd.read_parquet(self.data_path / "item_features.parquet")
        self.user_features = pd.read_parquet(self.data_path / "user_features.parquet")

        # Build product index mapping
        self.product_to_idx = {
            p: i for i, p in enumerate(self.product_encoder.classes_)
        }
        self.idx_to_product = {
            i: p for p, i in self.product_to_idx.items()
        }

        logger.info("Loaded %d products, %d users.", len(self.product_encoder.classes_), len(self.user_features))

    def _load_models(self) -> None:
        """Load trained UserTower and ProductTower."""
        user_checkpoint = self.model_path / "user_tower_checkpoint.pth"
        product_checkpoint = self.model_path / "product_tower_checkpoint.pth"

        if not user_checkpoint.exists() or not product_checkpoint.exists():
            raise FileNotFoundError(
                "Model checkpoints not found. Train the model first with `python main/train.py`."
            )

        logger.info("Loading models...")

        self.user_tower = UserTower(
            product_vocab_size=len(self.product_encoder.classes_),
            product_embed_dim=32,
            hidden_dim=128,
            num_layers=2,
            output_dim=64,
        ).to(self.device)
        user_ckpt = torch.load(user_checkpoint, map_location=self.device, weights_only=True)
        self.user_tower.load_state_dict(user_ckpt["model_state_dict"])
        self.user_tower.eval()

        self.product_tower = ProductTower(
            brand_vocab_size=len(self.brand_encoder.classes_),
            brand_embed_dim=16,
            hidden_dim=128,
            output_dim=64,
        ).to(self.device)
        product_ckpt = torch.load(product_checkpoint, map_location=self.device, weights_only=True)
        self.product_tower.load_state_dict(product_ckpt["model_state_dict"])
        self.product_tower.eval()

        logger.info("Models loaded on %s.", self.device)

    def _get_user_vector(self, user_history: list[int]) -> np.ndarray:
        """Compute user embedding from their interaction history."""
        history_tensor = torch.tensor(user_history, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            user_vec = self.user_tower(history_tensor.to(self.device))
        return user_vec.cpu().numpy().flatten()

    def _get_all_product_vectors(self) -> np.ndarray:
        """Precompute all product vectors for efficient scoring.
        
        Uses cached text embeddings if available, otherwise computes them once.
        """
        # Check for cached product vectors
        vectors_path = self.data_path / "product_vectors.npy"
        
        if vectors_path.exists():
            logger.info("Loading cached product vectors...")
            vectors = np.load(vectors_path)
            logger.info("Loaded cached vectors with shape: %s", vectors.shape)
            return vectors
        
        # No cache - compute once
        logger.info("Computing all product vectors (this may take a while)...")
        from sentence_transformers import SentenceTransformer
        text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        all_vectors = []
        batch_size = 512

        for i in range(0, len(self.item_features), batch_size):
            batch = self.item_features.iloc[i : i + batch_size]
            text_inputs = [
                f"{row['title']} {row['description']}"
                for _, row in batch.iterrows()
            ]

            text_embs = text_model.encode(text_inputs, convert_to_tensor=True, show_progress_bar=False)
            brand_ids = torch.tensor(batch["brand_id"].values, dtype=torch.long)
            prices = torch.zeros(len(batch), dtype=torch.float32)

            with torch.no_grad():
                product_vecs = self.product_tower(
                    text_embs.to(self.device),
                    brand_ids.to(self.device),
                    prices.to(self.device),
                )
            all_vectors.append(product_vecs.cpu().numpy())

        result = np.vstack(all_vectors)
        np.save(vectors_path, result)
        logger.info("Saved product vectors with shape: %s", result.shape)
        return result

    def _search_with_milvus(
        self, user_vector: np.ndarray, k: int = 50
    ) -> list[str]:
        """Search for top-K similar products using Milvus ANN.
        
        Args:
            user_vector: User embedding of shape (dim,).
            k: Number of results to return.
            
        Returns:
            List of product IDs ranked by similarity.
        """
        from pymilvus import Collection, connections, utility
        
        collection_name = "product_vectors"
        
        try:
            # Connect to Milvus
            connections.connect(
                host="localhost",
                port="19530",
            )
            
            if not utility.has_collection(collection_name):
                logger.warning("Milvus collection '%s' not found, falling back to brute-force", collection_name)
                return self._brute_force_search(user_vector, k)
            
            collection = Collection(collection_name)
            collection.load()
            
            # Search with inner product similarity
            search_params = {
                "metric_type": "IP",  # Inner Product
                "params": {"nprobe": 16},
            }
            
            results = collection.search(
                data=[user_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=k,
                output_fields=["product_id"],
            )
            
            # Extract product IDs
            product_ids = []
            for hits in results:
                for hit in hits:
                    product_ids.append(str(hit.entity.get("product_id")))
            
            return product_ids
            
        except Exception as exc:
            logger.warning("Milvus search failed (%s), falling back to brute-force", exc)
            return self._brute_force_search(user_vector, k)

    def _brute_force_search(
        self, user_vector: np.ndarray, k: int
    ) -> list[str]:
        """Fallback brute-force search when Milvus is unavailable."""
        product_vectors = self._get_all_product_vectors()
        scores = product_vectors @ user_vector
        ranked_indices = np.argsort(-scores)[:k]
        return [
            str(self.item_features.iloc[idx]["product_id"])
            for idx in ranked_indices
        ]

    def evaluate(
        self,
        k_values: list[int] | None = None,
        sample_users: int | None = None,
        use_milvus: bool = False,
    ) -> dict[str, Any]:
        """Run full evaluation and return metrics.

        Args:
            k_values: List of K values for @K metrics.
            sample_users: Number of users to evaluate (None for all).
            use_milvus: If True, use Milvus ANN search instead of brute-force.

        Returns:
            Dictionary with all computed metrics.
        """
        k_values = k_values or [5, 10, 20, 50]
        logger.info("=== STARTING EVALUATION ===")
        logger.info("K values: %s", k_values)
        logger.info("Search method: %s", "Milvus ANN" if use_milvus else "Brute-force")

        # Precompute product vectors (only needed for brute-force)
        if not use_milvus:
            product_vectors = self._get_all_product_vectors()
        else:
            product_vectors = None

        # Prepare test users
        test_users = self.user_features.copy()
        if sample_users:
            test_users = test_users.sample(n=min(sample_users, len(test_users)), random_state=42)

        # Metrics accumulators
        metrics: dict[str, list[float]] = {
            f"recall@{k}": [] for k in k_values
        }
        metrics.update({f"ndcg@{k}": [] for k in k_values})
        metrics.update({f"hit_rate@{k}": [] for k in k_values})
        metrics.update({f"mrr@{k}": [] for k in k_values})

        recommended_items_all: set[int] = set()
        max_k_needed = max(k_values)

        for _, user_row in tqdm(test_users.iterrows(), total=len(test_users), desc="Evaluating users"):
            history = user_row["history_padded"]

            # Remove padding
            actual_history = [int(p) for p in history if p != 0]

            if len(actual_history) < 3:
                continue

            # Use last 20% of history as test set
            split_idx = int(len(actual_history) * 0.8)
            train_history = actual_history[:split_idx]
            test_items = actual_history[split_idx:]

            if not test_items:
                continue

            # Get user vector
            user_vector = self._get_user_vector(train_history)

            # Search for recommendations - return INDICES not product IDs
            if use_milvus:
                ranked_indices = self._search_with_milvus(user_vector, k=max_k_needed)
            else:
                # Brute-force: score all products, return indices
                scores = product_vectors @ user_vector
                ranked_indices = np.argsort(-scores)[:max_k_needed]

            # Track recommended items for coverage (indices)
            if hasattr(ranked_indices, "__iter__"):
                recommended_items_all.update(list(ranked_indices)[:20])

            # Compute metrics - compare indices to indices
            for k in k_values:
                metrics[f"recall@{k}"].append(recall_at_k(list(ranked_indices), test_items, k))
                metrics[f"ndcg@{k}"].append(ndcg_at_k(list(ranked_indices), test_items, k))
                metrics[f"hit_rate@{k}"].append(hit_rate_at_k(list(ranked_indices), test_items, k))
                metrics[f"mrr@{k}"].append(mrr_at_k(list(ranked_indices), test_items, k))

        # Aggregate results
        results: dict[str, Any] = {}
        for metric_name, values in metrics.items():
            results[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
            }

        # Coverage
        total_products = len(self.product_encoder.classes_)
        results["item_coverage"] = len(recommended_items_all) / total_products

        # Summary
        results["num_users_evaluated"] = len(test_users)
        results["k_values"] = k_values

        self._print_results(results)
        return results

    def _print_results(self, results: dict[str, Any]) -> None:
        """Print evaluation results in a formatted table."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        k_values = results["k_values"]

        print(f"\n{'Metric':<15} {'K':>3} {'Mean':>8} {'Std':>8} {'Median':>8}")
        print("-" * 50)

        for metric_prefix in ["recall", "ndcg", "hit_rate", "mrr"]:
            for k in k_values:
                key = f"{metric_prefix}@{k}"
                stats = results[key]
                print(
                    f"{metric_prefix:<15} {k:>3} "
                    f"{stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['median']:>8.4f}"
                )

        print("-" * 50)
        print(f"{'Item Coverage':<15} {'':>3} {results['item_coverage']:>8.4f}")
        print(f"{'Users Evaluated':<15} {'':>3} {results['num_users_evaluated']:>8}")
        print("=" * 60)


def main() -> None:
    """Run evaluation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Two-Tower recommendation model")
    parser.add_argument("--data-dir", default="data/processed_data", help="Path to processed data")
    parser.add_argument("--model-dir", default="model", help="Path to model checkpoints")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20, 50], help="K values for @K metrics")
    parser.add_argument("--sample-users", type=int, default=None, help="Number of users to evaluate (None for all)")
    parser.add_argument("--device", default=None, help="Torch device (auto-detected if not specified)")
    parser.add_argument("--use-milvus", action="store_true", help="Use Milvus ANN search (much faster)")

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        device=args.device,
    )

    results = evaluator.evaluate(
        k_values=args.k_values,
        sample_users=args.sample_users,
        use_milvus=args.use_milvus,
    )

    # Save results
    output_path = Path(args.model_dir) / "evaluation_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()