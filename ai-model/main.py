"""Content-based filtering pipeline with Milvus ANN search.

Pipeline: SentenceTransformer embedding → Milvus IVF_FLAT → Precision@K/Recall@K/NDCG

Usage::

    python main.py                           # full pipeline
    python main.py --mode preprocess         # embeddings + split
    python main.py --mode ingest             # insert into Milvus
    python main.py --mode evaluate           # Milvus search + metrics
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from src.evaluate import print_evaluation, evaluate_content_model
from src.model import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    ingest_vectors,
    init_milvus_collection,
    search_similar_batch,
)
from src.preprocess import (
    build_text_features,
    compute_embeddings,
    create_content_train_test,
    load_processed_data,
    load_products,
    save_processed_data,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── Pipeline steps ────────────────────────────────────────────────────


def run_preprocess(args: argparse.Namespace) -> None:
    """Preprocess products → SentenceTransformer embeddings → train/test split.

    Args:
        args: Parsed CLI arguments.
    """
    logger.info("=== Preprocessing Pipeline ===")

    products = load_products(args.csv_path)
    products_text = build_text_features(products)

    dense_vectors, model = compute_embeddings(
        products_text["text_content"].tolist(),
    )

    train_products, test_queries, train_vectors, test_vectors, ground_truth = create_content_train_test(
        products_text,
        dense_vectors,
        test_size=args.test_size,
        n_queries=args.n_queries,
        random_state=args.seed,
    )

    save_processed_data(
        train_products, test_queries, products_text,
        train_vectors, test_vectors, ground_truth, args.output_dir,
    )

    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"  Products:     {len(products_text)}")
    print(f"  Train:        {len(train_products)}")
    print(f"  Test queries: {len(test_queries)}")
    print(f"  Embedding:    {dense_vectors.shape[1]}-dim (SentenceTransformer)")
    print(f"  Output dir:   {args.output_dir}")


def run_ingest(args: argparse.Namespace) -> None:
    """Bulk-insert train vectors into Milvus.

    Args:
        args: Parsed CLI arguments.
    """
    logger.info("=== Ingesting vectors into Milvus ===")

    data = load_processed_data(args.data_dir)
    train_vectors = data["train_vectors"]
    train_products = data["train_products"]

    product_ids = train_products.get("asin", train_products.index.astype(str)).tolist()
    product_idxs = list(range(len(train_vectors)))
    brands = train_products.get("brand", pd.Series([""] * len(train_products))).tolist()

    logger.info("Initialising Milvus collection (dim=%d)…", EMBEDDING_DIM)
    init_milvus_collection(dim=EMBEDDING_DIM)
    logger.info("Bulk-inserting %d vectors…", len(train_vectors))
    ingest_vectors(train_vectors, product_ids, product_idxs, brands)

    print("\n=== INGESTION COMPLETE ===")
    print(f"  Vectors ingested: {len(train_vectors)}")
    print(f"  Collection:       {COLLECTION_NAME}")


def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate content-based retrieval via Milvus ANN search.

    Each test query vector is searched in Milvus; predictions are compared
    against brand-based ground truth.

    Args:
        args: Parsed CLI arguments.
    """
    logger.info("=== Evaluation Pipeline ===")

    data = load_processed_data(args.data_dir)
    test_queries = data["test_queries"]
    test_vectors = data["test_vectors"]
    ground_truth = data["ground_truth"]
    k_values = args.k_values

    logger.info("Running Milvus batch search for %d queries (top_k=%d)…", len(test_vectors), max(k_values))
    milvus_results = search_similar_batch(test_vectors, top_k=max(k_values))

    results = evaluate_content_model(ground_truth, milvus_results, k_values=k_values)
    print_evaluation(results)

    if args.save_results:
        results_path = Path(args.output_dir) / "evaluation_results_milvus.pkl"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        logger.info("Results saved to %s", results_path)


def run_full(args: argparse.Namespace) -> None:
    """Run the complete pipeline: preprocess → ingest → evaluate.

    Args:
        args: Parsed CLI arguments.
    """
    run_preprocess(args)
    print()
    run_ingest(args)
    print()
    run_evaluate(args)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Content-Based Filtering with Milvus")

    parser.add_argument(
        "--mode", default="full",
        choices=["preprocess", "ingest", "evaluate", "full"],
        help="Pipeline stage",
    )
    parser.add_argument("--csv-path", default="data/dataset/products_clean.csv")
    parser.add_argument("--output-dir", default="data/processed_data")
    parser.add_argument("--data-dir", default="data/processed_data")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--gt-top-k", type=int, default=50, help="Ground truth top-K for semantic matching")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--save-results", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    modes = {
        "preprocess": run_preprocess,
        "ingest": run_ingest,
        "evaluate": run_evaluate,
        "full": run_full,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    import pandas as pd
    main()
