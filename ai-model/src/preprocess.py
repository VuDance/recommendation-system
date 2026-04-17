"""Data preprocessing module for content-based filtering.

Uses SentenceTransformer to encode product metadata into semantic dense
vectors, then creates train/test splits with brand-based ground truth.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 128

DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_MIN_GROUND_TRUTH: int = 2


# ── Data loading ─────────────────────────────────────────────────────


def load_products(csv_path: str | Path) -> pd.DataFrame:
    """Load the cleaned products dataset.

    Args:
        csv_path: Path to product_clean.csv.

    Returns:
        DataFrame with product metadata columns.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    logger.info("Loading products from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d products with columns: %s", len(df), list(df.columns))
    return df


# ── Text feature construction ────────────────────────────────────────


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine product metadata into a single semantic text field.

    Args:
        df: Products DataFrame.

    Returns:
        DataFrame with an added *text_content* column.
    """
    products = df.copy()

    def combine_text(row: pd.Series) -> str:
        parts: list[str] = []
        for col in ["title", "description", "brand"]:
            if col not in row.index:
                continue
            val = str(row[col]).strip()
            if val and val != "nan":
                parts.append(val)
        return " ".join(parts)

    products["text_content"] = products.apply(combine_text, axis=1)
    products = products[products["text_content"].str.len() > 0].reset_index(drop=True)

    logger.info("Built text features for %d products", len(products))
    return products


# ── SentenceTransformer embeddings ───────────────────────────────────


def compute_embeddings(
    texts: list[str],
    show_progress: bool = True,
) -> tuple[np.ndarray, SentenceTransformer]:
    """Encode product text into dense 384-dim embeddings via SentenceTransformer.

    Milvus stores these dense vectors.  They are L2-normalised so that
    the Inner Product metric in Milvus is equivalent to cosine similarity.

    Args:
        texts: List of product text strings.
        show_progress: Whether to display a progress bar.

    Returns:
        Tuple of (dense_vectors L2-normalised, fitted SentenceTransformer model).
    """
    logger.info("Encoding %d texts with SentenceTransformer '%s'", len(texts), MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)
    dense_vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )

    # L2-normalise so IP ≈ cosine in Milvus
    norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    dense_vectors = dense_vectors / norms

    logger.info("Embeddings shape: %s", dense_vectors.shape)
    return dense_vectors, model


# ── Train / test split with semantic ground truth ─────────────────────


def build_semantic_ground_truth(
    test_vectors: np.ndarray,
    train_vectors: np.ndarray,
    top_k: int = 50,
) -> tuple[pd.DataFrame, np.ndarray, list[list[int]]]:
    """Build ground truth via exact cosine similarity on the embedding space.

    For each test query, finds the most similar train products via exact
    brute-force cosine similarity (equivalent to IP on L2-normalised vectors).
    Milvus ANN results are then compared against these as ground truth.

    Args:
        test_vectors: Test embedding matrix (n_test, dim), L2-normalised.
        train_vectors: Train embedding matrix (n_train, dim), L2-normalised.
        top_k: Number of ground truth items per query.

    Returns:
        Tuple of (test_queries_df, exact_similarities, ground_truth_indices).
    """
    from tqdm import tqdm

    logger.info("Building semantic ground truth for %d queries (top_k=%d)…", len(test_vectors), top_k)

    # Compute exact cosine similarity: test @ train.T  (L2-normalized → IP = cosine)
    exact_sims = test_vectors @ train_vectors.T  # (n_test, n_train)

    # Get top-k ground truth indices
    ground_truth = np.argsort(-exact_sims, axis=1)[:, :top_k]
    gt_lists = ground_truth.tolist()

    # Filter out test queries that only see themselves (if a test product leaked into train)
    test_indices = list(range(len(test_vectors)))
    valid_mask = []
    for i, gt in enumerate(gt_lists):
        # Ensure we have enough distinct train items
        valid_mask.append(len(gt) >= 2)

    gt_lists_filtered = [gt for gt, valid in zip(gt_lists, valid_mask) if valid]

    logger.info(
        "Ground truth built — %d valid queries, avg ground-truth size: %.1f",
        len(gt_lists_filtered),
        np.mean([len(g) for g in gt_lists_filtered]),
    )

    # Compute avg cosine similarity of ground truth as a sanity check
    gt_sims = []
    for i, gt in enumerate(gt_lists):  # use original gt_lists
        if len(gt) > 0:
            avg_sim = np.mean(exact_sims[i, gt[:top_k]])
            gt_sims.append(avg_sim)

    logger.info("Avg cosine similarity of GT items: %.4f", np.mean(gt_sims))

    test_queries_df = pd.DataFrame(
        [
            {
                "query_index": i,
                "ground_truth_ids": gt,
                "n_gt_items": len(gt),
            }
            for i, gt in zip(
                range(len(gt_lists)),
                gt_lists,
            )
            if len(gt) >= 2
        ]
    )

    return test_queries_df, exact_sims, gt_lists_filtered


def create_content_train_test(
    products_df: pd.DataFrame,
    dense_vectors: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    n_queries: int | None = None,
    top_k: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[list[int]]]:
    """Split products into train/test sets for content-based evaluation.

    Ground truth is the set of most similar train products via exact
    cosine similarity on the SentenceTransformer embeddings.

    Args:
        products_df: Products DataFrame with text_content, brand, product_id.
        dense_vectors: L2-normalised embeddings (same row order as products_df).
        test_size: Fraction of products reserved as queries.
        n_queries: Upper bound on test queries.
        top_k: Number of ground truth items per query.
        random_state: RNG seed.

    Returns:
        Tuple of (train_products, test_queries_df, train_vectors, test_vectors, ground_truth).
    """
    logger.info("Creating train/test split (test_size=%.2f)", test_size)

    train_idx, test_idx = train_test_split(
        np.arange(len(products_df)), test_size=test_size, random_state=random_state
    )
    if n_queries:
        test_idx = test_idx[:n_queries]

    train_products = products_df.iloc[train_idx].reset_index(drop=True)
    test_products = products_df.iloc[test_idx].reset_index(drop=True)
    train_vectors = dense_vectors[train_idx]
    test_vectors = dense_vectors[test_idx]

    # ── Build semantic ground truth via exact cosine similarity ───────
    test_queries_df, exact_sims, gt_lists = build_semantic_ground_truth(
        test_vectors, train_vectors, top_k=top_k,
    )

    logger.info(
        "Split complete — train=%d, test_queries=%d",
        len(train_products),
        len(test_queries_df),
    )
    return train_products, test_queries_df, train_vectors, test_vectors, gt_lists


# ── Persistence ───────────────────────────────────────────────────────


def save_processed_data(
    train_products: pd.DataFrame,
    test_queries_df: pd.DataFrame,
    all_products_df: pd.DataFrame,
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
    ground_truth: list[list[int]],
    output_dir: str | Path,
) -> None:
    """Save processed data and embeddings to disk.

    Args:
        train_products: Train split products.
        test_queries_df: Test queries with ground truth.
        all_products_df: Full product catalog.
        train_vectors: Train embeddings (L2-normalised).
        test_vectors: Test embeddings (L2-normalised).
        ground_truth: Semantic ground truth train indices per query.
        output_dir: Output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_products.to_parquet(out / "content_train_products.parquet", index=False)
    test_queries_df.to_parquet(out / "content_test_queries.parquet", index=False)
    all_products_df.to_parquet(out / "content_products.parquet", index=False)

    np.save(out / "content_train_vectors.npy", train_vectors)
    np.save(out / "content_test_vectors.npy", test_vectors)

    with open(out / "content_ground_truth.pkl", "wb") as f:
        pickle.dump(ground_truth, f)

    metadata = {
        "num_train": len(train_products),
        "num_test_queries": len(test_queries_df),
        "num_products": len(all_products_df),
        "embedding_dim": train_vectors.shape[1],
        "model": MODEL_NAME,
    }
    with open(out / "content_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    logger.info("Processed data saved to %s", out)


def load_processed_data(output_dir: str | Path) -> dict[str, Any]:
    """Load processed data from disk.

    Args:
        output_dir: Directory containing processed artefacts.

    Returns:
        Dict with keys: train_products, test_queries, all_products,
        train_vectors, test_vectors, ground_truth, metadata.
    """
    out = Path(output_dir)

    train_products = pd.read_parquet(out / "content_train_products.parquet")
    test_queries = pd.read_parquet(out / "content_test_queries.parquet")
    all_products = pd.read_parquet(out / "content_products.parquet")
    train_vectors = np.load(out / "content_train_vectors.npy")
    test_vectors = np.load(out / "content_test_vectors.npy")

    with open(out / "content_ground_truth.pkl", "rb") as f:
        ground_truth = pickle.load(f)
    with open(out / "content_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return {
        "train_products": train_products,
        "test_queries": test_queries,
        "all_products": all_products,
        "train_vectors": train_vectors,
        "test_vectors": test_vectors,
        "ground_truth": ground_truth,
        "metadata": metadata,
    }
