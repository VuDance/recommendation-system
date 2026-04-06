"""Evaluation module for content-based filtering using text embeddings.

Implements standard recommendation system metrics for content-based filtering:
- Recall@K, NDCG@K, Hit Rate@K, MRR@K
- Item coverage and popularity bias
- Per-user and aggregate statistics

This module evaluates content-based recommendations by:
1. Encoding query text from the test set
2. Computing similarity against all product embeddings
3. Comparing results against ground truth similar products
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -- Evaluation Metrics --


def recall_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Fraction of relevant items that appear in top-K recommendations."""
    if not ground_truth:
        return 0.0
    relevant_in_topk = len(set(retrieved[:k]) & set(ground_truth))
    return relevant_in_topk / len(ground_truth)


def ndcg_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not ground_truth:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Whether at least one relevant item appears in top-K."""
    return 1.0 if set(retrieved[:k]) & set(ground_truth) else 0.0


def mrr_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Mean Reciprocal Rank at K."""
    for i, item in enumerate(retrieved[:k]):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


# -- Content-Based Evaluation Pipeline --


class ContentBasedEvaluator:
    """Evaluates content-based filtering using pre-computed embeddings.

    Args:
        test_data_dir: Path to directory containing test set files.
        embedding_model: SentenceTransformer model name for encoding.
    """

    def __init__(
        self,
        test_data_dir: str = "data/processed_data",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.data_path = Path(test_data_dir)

        # Load test data
        self._load_test_data()

        # Load embeddings
        self._load_embeddings()

        # Initialize encoder
        self.text_model = SentenceTransformer(embedding_model)

    def _load_test_data(self) -> None:
        """Load test queries and products from disk."""
        logger.info("Loading test data...")

        test_queries_path = self.data_path / "content_test_queries.parquet"
        products_path = self.data_path / "content_products.parquet"
        metadata_path = self.data_path / "content_test_metadata.pkl"

        if not test_queries_path.exists():
            raise FileNotFoundError(
                f"Test queries file not found: {test_queries_path}\n"
                "Run generate_content_test_set.py first."
            )

        self.test_queries = pd.read_parquet(test_queries_path)

        # Ensure ground_truth_ids is stored as a proper Python list
        def _ensure_list(val):
            if isinstance(val, list):
                return [int(x) for x in val]
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, str):
                import ast
                try:
                    parsed = ast.literal_eval(val)
                    return [int(x) for x in parsed]
                except (ValueError, SyntaxError):
                    return []
            return []

        self.test_queries["ground_truth_ids"] = self.test_queries["ground_truth_ids"].apply(_ensure_list)
        self.test_queries["product_index"] = self.test_queries["product_index"].astype(int)

        logger.info("Loaded %d test queries", len(self.test_queries))

        if products_path.exists():
            self.products = pd.read_parquet(products_path)
            logger.info("Loaded %d products", len(self.products))
        else:
            self.products = None

        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}

    def _load_embeddings(self) -> None:
        """Load pre-computed embeddings."""
        emb_path = self.data_path / "content_embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {emb_path}\n"
                "Run generate_content_test_set.py first."
            )
        self.embeddings = np.load(emb_path)
        logger.info("Loaded embeddings: shape = %s", self.embeddings.shape)

    def _search(
        self, query_text: str, query_index: int, k: int = 50
    ) -> list[int]:
        """Encode query and find most similar products via brute-force cosine similarity.

        Args:
            query_text: Text to encode and search for.
            query_index: Index of the query product (excluded from results).
            k: Number of results to return.

        Returns:
            List of product indices (int).
        """
        query_vector = self.text_model.encode(query_text, convert_to_numpy=True)

        # Cosine similarity
        similarities = self.embeddings @ query_vector / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector) + 1e-8
        )

        # Exclude self
        similarities[query_index] = -1.0

        # Get top-k indices
        top_k = np.argsort(-similarities)[:k].tolist()
        return [int(i) for i in top_k]

    def evaluate(
        self,
        k_values: list[int] | None = None,
        sample_queries: int | None = None,
    ) -> dict[str, Any]:
        """Run full evaluation and return metrics.

        Args:
            k_values: List of K values for @K metrics.
            sample_queries: Number of queries to evaluate (None for all).

        Returns:
            Dictionary with all computed metrics.
        """
        k_values = k_values or [5, 10, 20, 50]
        logger.info("=== STARTING CONTENT-BASED EVALUATION ===")
        logger.info("K values: %s", k_values)

        # Prepare test queries
        test_queries = self.test_queries.copy()
        if sample_queries:
            test_queries = test_queries.sample(
                n=min(sample_queries, len(test_queries)), random_state=42
            )

        # Metrics accumulators
        metrics: dict[str, list[float]] = {f"recall@{k}": [] for k in k_values}
        metrics.update({f"ndcg@{k}": [] for k in k_values})
        metrics.update({f"hit_rate@{k}": [] for k in k_values})
        metrics.update({f"mrr@{k}": [] for k in k_values})

        recommended_items_all: set[int] = set()
        max_k = max(k_values)
        n_evaluated = 0

        for _, row in tqdm(test_queries.iterrows(), total=len(test_queries), desc="Evaluating"):
            query_text = str(row["query_text"])
            query_index = int(row["product_index"])
            ground_truth: list[int] = list(row["ground_truth_ids"])

            if not ground_truth:
                continue

            # Search
            retrieved = self._search(query_text, query_index, k=max_k)

            n_evaluated += 1
            recommended_items_all.update(retrieved[:20])

            # Compute metrics
            for k in k_values:
                metrics[f"recall@{k}"].append(recall_at_k(retrieved, ground_truth, k))
                metrics[f"ndcg@{k}"].append(ndcg_at_k(retrieved, ground_truth, k))
                metrics[f"hit_rate@{k}"].append(hit_rate_at_k(retrieved, ground_truth, k))
                metrics[f"mrr@{k}"].append(mrr_at_k(retrieved, ground_truth, k))

        if n_evaluated == 0:
            logger.warning("No queries were evaluated. Check test data format.")
            return {}

        # Aggregate results
        results: dict[str, Any] = {}
        for metric_name, values in metrics.items():
            results[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
            }

        total_products = len(self.products) if self.products is not None else 0
        results["item_coverage"] = (
            len(recommended_items_all) / total_products if total_products > 0 else 0.0
        )
        results["num_queries_evaluated"] = n_evaluated
        results["k_values"] = k_values

        self._print_results(results)
        return results

    def _print_results(self, results: dict[str, Any]) -> None:
        """Print evaluation results in a formatted table."""
        print("\n" + "=" * 60)
        print("CONTENT-BASED FILTERING EVALUATION RESULTS")
        print("=" * 60)

        k_values = results["k_values"]

        print(f"\n{'Metric':<15} {'K':>3} {'Mean':>8} {'Std':>8} {'Median':>8}")
        print("-" * 50)

        for prefix in ["recall", "ndcg", "hit_rate", "mrr"]:
            for k in k_values:
                key = f"{prefix}@{k}"
                stats = results[key]
                print(
                    f"{prefix:<15} {k:>3} "
                    f"{stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['median']:>8.4f}"
                )

        print("-" * 50)
        print(f"{'Item Coverage':<15} {'':>3} {results['item_coverage']:>8.4f}")
        print(f"{'Queries Eval.':<15} {'':>3} {results['num_queries_evaluated']:>8}")
        print("=" * 60)


def main() -> None:
    """Run evaluation from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate content-based filtering model"
    )
    parser.add_argument(
        "--data-dir", default="data/processed_data", help="Path to test data"
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50],
        help="K values for @K metrics",
    )
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=None,
        help="Number of queries to evaluate (None for all)",
    )

    args = parser.parse_args()

    evaluator = ContentBasedEvaluator(
        test_data_dir=args.data_dir,
    )

    results = evaluator.evaluate(
        k_values=args.k_values,
        sample_queries=args.sample_queries,
    )

    # Save results
    output_path = Path(args.data_dir) / "content_evaluation_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()