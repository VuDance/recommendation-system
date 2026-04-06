"""Generate test set for content-based filtering evaluation from products_clean.csv.

This module creates query-product pairs for evaluating content-based filtering.
For each product, it generates a search query from its title/description and
expects similar products (same brand or category) as ground truth.
"""

import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MIN_RELEVANT_ITEMS = 3
MAX_TEST_QUERIES = 1000


def load_products_clean(csv_path: str | Path) -> pd.DataFrame:
    """Load the cleaned products CSV.

    Args:
        csv_path: Path to products_clean.csv.

    Returns:
        DataFrame with cleaned product data.
    """
    logger.info("Loading products_clean.csv from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d products with columns: %s", len(df), list(df.columns))
    return df


def extract_brand_from_title(title: str) -> str:
    """Extract brand name from product title (heuristic).

    Args:
        title: Product title string.

    Returns:
        Extracted brand name or empty string.
    """
    if pd.isna(title) or not isinstance(title, str):
        return ""
    
    # Common brand patterns - take first 1-2 words as potential brand
    words = title.strip().split()
    if len(words) >= 2:
        # Check if title starts with known patterns
        brand = words[0]
        # Skip short/generic words
        if len(brand) > 2 and brand.lower() not in ["the", "and", "for", "with", "men", "women", "womens"]:
            return brand
    return ""


def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clean feature columns for products.

    Args:
        df: Raw products DataFrame.

    Returns:
        DataFrame with standardized columns: product_id, title, brand, text_content.
    """
    products = df.copy()
    
    # Standardize column names
    if "asin" in products.columns:
        products["product_id"] = products["asin"]
    elif "product_id" not in products.columns:
        products["product_id"] = range(len(products))
    
    # Extract brand if not present
    if "brand" not in products.columns or products["brand"].isna().all():
        products["brand"] = products["title"].apply(extract_brand_from_title)
    else:
        products["brand"] = products["brand"].fillna("")
    
    # Create combined text for embedding
    def combine_text(row):
        parts = []
        if pd.notna(row.get("title", "")):
            parts.append(str(row["title"]))
        if pd.notna(row.get("description", "")):
            desc = str(row["description"])
            if desc and desc != "nan" and len(desc) > 10:
                parts.append(desc)
        if pd.notna(row.get("brand", "")):
            brand = str(row["brand"])
            if brand and brand != "nan":
                parts.append(brand)
        return " ".join(parts)
    
    products["text_content"] = products.apply(combine_text, axis=1)
    
    # Filter out products with empty content
    products = products[products["text_content"].str.len() > 10].copy()
    
    logger.info("Processed %d products with valid content", len(products))
    return products


def find_similar_products(
    product_idx: int,
    products_df: pd.DataFrame,
    embeddings: np.ndarray,
    brand: str,
    min_similar: int = DEFAULT_MIN_RELEVANT_ITEMS,
    top_k: int = 50,
) -> list[int]:
    """Find similar products based on embedding similarity and brand match.

    Args:
        product_idx: Index of the query product in embeddings array.
        products_df: DataFrame with product metadata.
        embeddings: Product embeddings array.
        brand: Brand of the query product.
        min_similar: Minimum number of similar products required.
        top_k: How many similar products to consider.

    Returns:
        List of indices of similar products that share the same brand.
    """
    query_vec = embeddings[product_idx]
    
    # Compute cosine similarity
    similarities = embeddings @ query_vec / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )
    
    # Get top-k most similar (excluding self)
    top_indices = np.argsort(-similarities)[1:top_k+1]
    
    # Filter by brand match
    similar_products = []
    for idx in top_indices:
        if idx < len(products_df):
            other_brand = products_df.iloc[idx].get("brand", "")
            if brand and other_brand and brand.lower() == other_brand.lower():
                similar_products.append(int(idx))
    
    return similar_products


def generate_test_queries(
    products_df: pd.DataFrame,
    n_test_queries: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate test queries for content-based filtering evaluation.

    For each test query, we create a search query from the product's
    text content and find ground-truth similar products.

    Args:
        products_df: Product features DataFrame.
        n_test_queries: Number of test queries to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns: query_text, query_product_id, ground_truth_ids,
        query_brand, product_index.
    """
    logger.info("Generating test queries...")
    
    # Compute embeddings
    logger.info("Computing text embeddings for %d products...", len(products_df))
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = text_model.encode(
        products_df["text_content"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    logger.info("Embeddings computed: shape = %s", embeddings.shape)
    
    # Save embeddings for later use
    embeddings_path = Path(__file__).parent / "processed_data" / "content_embeddings.npy"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    logger.info("Embeddings saved to %s", embeddings_path)
    
    # Find suitable test products (products with similar items)
    logger.info("Finding similar products for ground truth...")
    test_data = []
    
    # Sample products to test
    n_to_test = n_test_queries or min(MAX_TEST_QUERIES, len(products_df))
    all_indices = list(range(len(products_df)))
    sampled_indices = random.sample(all_indices, min(n_to_test * 3, len(all_indices)))
    
    for product_idx in tqdm(sampled_indices, desc="Finding similar products"):
        product = products_df.iloc[product_idx]
        brand = product.get("brand", "")
        
        similar_products = find_similar_products(
            product_idx, products_df, embeddings, brand
        )
        
        if len(similar_products) >= DEFAULT_MIN_RELEVANT_ITEMS:
            query_text = product["text_content"]
            
            test_data.append({
                "query_text": query_text,
                "query_product_id": product["product_id"],
                "ground_truth_ids": similar_products[:20],  # Limit to 20 ground truth
                "query_brand": brand,
                "product_index": int(product_idx),
            })
        
        if len(test_data) >= n_to_test:
            break
    
    test_df = pd.DataFrame(test_data)
    logger.info("Generated %d test queries with valid ground truth", len(test_df))
    
    # Summary statistics
    if len(test_df) > 0:
        avg_gt = test_df["ground_truth_ids"].apply(len).mean()
        logger.info("Average ground truth items per query: %.1f", avg_gt)
    
    return test_df


def save_test_set(
    test_df: pd.DataFrame,
    products_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save test set to disk.

    Args:
        test_df: Test queries DataFrame.
        products_df: Full products DataFrame.
        output_dir: Directory to save test set files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save test queries
    test_df.to_parquet(output_path / "content_test_queries.parquet", index=False)
    
    # Save products reference
    products_df.to_parquet(output_path / "content_products.parquet", index=False)
    
    # Save test set metadata
    metadata = {
        "num_test_queries": len(test_df),
        "num_products": len(products_df),
        "avg_ground_truth_size": float(test_df["ground_truth_ids"].apply(len).mean()),
        "ground_truth_cols": ["ground_truth_ids"],
    }
    with open(output_path / "content_test_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    logger.info("Test set saved to %s", output_path)
    logger.info("Test queries: %d", metadata["num_test_queries"])
    logger.info("Products in catalog: %d", metadata["num_products"])
    logger.info("Avg ground truth size: %.1f", metadata["avg_ground_truth_size"])


def main() -> None:
    """Run the test set generation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test set for content-based filtering")
    parser.add_argument(
        "--csv-path",
        default="data/dataset/products_clean.csv",
        help="Path to products_clean.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_data",
        help="Directory to save test set",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=1000,
        help="Number of test queries to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Load products
    products = load_products_clean(args.csv_path)
    
    # Create features
    products = create_product_features(products)
    
    # Generate test queries
    test_queries = generate_test_queries(
        products,
        n_test_queries=args.n_queries,
        random_state=args.seed,
    )
    
    # Save everything
    save_test_set(test_queries, products, args.output_dir)
    
    print("\n=== TEST SET GENERATION COMPLETE ===")
    print(f"Products processed: {len(products)}")
    print(f"Test queries generated: {len(test_queries)}")
    print(f"Average ground truth per query: {test_queries['ground_truth_ids'].apply(len).mean():.1f}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()