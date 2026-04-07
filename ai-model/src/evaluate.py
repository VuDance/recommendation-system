"""Evaluation metrics and pipeline for content-based filtering."""

import logging
from dataclasses import dataclass, field

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation metrics at multiple K thresholds."""

    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    catalog_coverage: float = 0.0
    avg_similarity: float = 0.0


# ── Metrics ──────────────────────────────────────────────────────────


def _mean_metric(
    predictions: list[list[int]],
    ground_truth: list[list[int]],
    metric_fn,
) -> float:
    """Compute the mean of *metric_fn* over all queries.

    Args:
        predictions: Predicted item indices per query.
        ground_truth: Ground-truth item indices per query.
        metric_fn: Callable(predictions_i, ground_truth_i, k) → float.

    Returns:
        Mean metric value.
    """
    if not predictions:
        return 0.0
    return float(np.mean([metric_fn(p, g) for p, g in zip(predictions, ground_truth)]))


def precision_at_k(
    predictions: list[list[int]],
    ground_truth: list[list[int]],
    k: int,
) -> float:
    """Mean Precision@K."""
    def _single(pred, gt):
        relevant = sum(1 for item in pred[:k] if item in gt)
        return relevant / k

    return _mean_metric(predictions, ground_truth, _single)


def recall_at_k(
    predictions: list[list[int]],
    ground_truth: list[list[int]],
    k: int,
) -> float:
    """Mean Recall@K."""
    def _single(pred, gt):
        if not gt:
            return 0.0
        relevant = sum(1 for item in pred[:k] if item in gt)
        return relevant / len(gt)

    return _mean_metric(predictions, ground_truth, _single)


def ndcg_at_k(
    predictions: list[list[int]],
    ground_truth: list[list[int]],
    k: int,
) -> float:
    """Mean NDCG@K (binary relevance)."""
    def _single(pred, gt):
        gt_set = set(gt)
        dcg = sum(
            1.0 / np.log2(rank + 2) for rank, item in enumerate(pred[:k]) if item in gt_set
        )
        ideal = min(k, len(gt))
        idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    return _mean_metric(predictions, ground_truth, _single)


def mean_reciprocal_rank(
    predictions: list[list[int]],
    ground_truth: list[list[int]],
) -> float:
    """Mean Reciprocal Rank."""
    rr_scores = []
    for pred, gt in zip(predictions, ground_truth):
        gt_set = set(gt)
        for rank, item in enumerate(pred, 1):
            if item in gt_set:
                rr_scores.append(1.0 / rank)
                break
        else:
            rr_scores.append(0.0)
    return float(np.mean(rr_scores))


# ── Orchestration ────────────────────────────────────────────────────


def evaluate_content_model(
    ground_truth_ids: list[list[int]],
    milvus_predictions: list[list[dict]],
    k_values: list[int] | None = None,
) -> EvaluationResults:
    """Evaluate content-based retrieval using Milvus search results.

    Args:
        ground_truth_ids: Ground-truth train indices per query.
        milvus_predictions: Milvus search results per query — each a list of
            dicts with at least a ``product_idx`` field.
        k_values: K thresholds.

    Returns:
        EvaluationResults with all metrics.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]

    # Convert Milvus dicts → plain index lists
    predicted_indices = [
        [int(hit["product_idx"]) for hit in recs] for recs in milvus_predictions
    ]

    # Ensure ground truth is plain list of lists of ints (may be numpy arrays)
    gt_lists = [
        [int(x) for x in (g.tolist() if hasattr(g, "tolist") else g)]
        for g in ground_truth_ids
    ]

    logger.info("Evaluating with K values: %s | queries=%d", k_values, len(predicted_indices))

    results = EvaluationResults()

    for k in k_values:
        p = precision_at_k(predicted_indices, gt_lists, k)
        r = recall_at_k(predicted_indices, gt_lists, k)
        n = ndcg_at_k(predicted_indices, gt_lists, k)
        results.precision_at_k[k] = p
        results.recall_at_k[k] = r
        results.ndcg_at_k[k] = n
        logger.info("Precision@%d: %.4f | Recall@%d: %.4f | NDCG@%d: %.4f", k, p, k, r, k, n)

    results.mrr = mean_reciprocal_rank(predicted_indices, gt_lists)
    logger.info("MRR: %.4f", results.mrr)

    # Catalog coverage — fraction of corpus that appears in any top-50 recommendation
    all_recommended: set[int] = set()
    for pred in predicted_indices:
        all_recommended.update(pred[:50])
    if predicted_indices:
        max_idx = max(max(p) for p in predicted_indices if p)
        results.catalog_coverage = len(all_recommended) / (max_idx + 1)
    logger.info("Catalog coverage (top-50): %.4f", results.catalog_coverage)

    # Average similarity score of the top-10 hit
    all_scores = []
    for recs in milvus_predictions:
        for hit in recs[:10]:
            all_scores.append(hit.get("score", 0.0))
    results.avg_similarity = float(np.mean(all_scores)) if all_scores else 0.0
    logger.info("Avg Milvus similarity (top-10): %.4f", results.avg_similarity)

    return results


def print_evaluation(results: EvaluationResults) -> None:
    """Pretty-print evaluation results."""
    ks = sorted(results.precision_at_k.keys())
    print("\n===== EVALUATION RESULTS (Milvus ANN) =====")
    for k in ks:
        print(f"  Precision@{k:2d}: {results.precision_at_k[k]:.4f}")
        print(f"  Recall@{k:2d}:    {results.recall_at_k[k]:.4f}")
        print(f"  NDCG@{k:2d}:      {results.ndcg_at_k[k]:.4f}")
    print(f"  MRR:               {results.mrr:.4f}")
    print(f"  Catalog coverage:  {results.catalog_coverage:.4f}")
    print(f"  Avg similarity:    {results.avg_similarity:.4f}")
    print("=============================================\n")
