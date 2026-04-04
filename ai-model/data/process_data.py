"""Data processing script for Two-Tower recommendation model.

Processes the Amazon Fashion dataset (reviews + metadata) into
clean parquet files and fitted label encoders ready for training.
"""

import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────

DEFAULT_MAX_HISTORY_LENGTH: int = 50
DEFAULT_MIN_PURCHASES: int = 3


# ── Data Loading ─────────────────────────────────────────


def load_jsonl_gz(filepath: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load a gzipped JSONL file into a DataFrame.

    Args:
        filepath: Path to the .json.gz file.
        max_rows: Maximum number of lines to read (None for unlimited).

    Returns:
        DataFrame with one row per successfully parsed JSON object.
    """
    logger.info("Loading data from %s", filepath)
    samples: list[dict] = []

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Error parsing line %d: %s", i, exc)

    logger.info("Loaded %d records.", len(samples))
    return pd.DataFrame(samples)


# ── Feature Processing ──────────────────────────────────


def process_item_features(product_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and clean item features from product metadata.

    Args:
        product_df: Raw product metadata DataFrame.

    Returns:
        Cleaned DataFrame with columns: asin, title, brand, description.
    """
    logger.info("Processing item features...")
    item_features = product_df[["asin", "title", "brand", "feature"]].copy()

    item_features["title"] = item_features["title"].fillna("")
    item_features["brand"] = item_features["brand"].fillna("Unknown")
    item_features["feature"] = item_features["feature"].fillna("")

    def extract_description(features: Any) -> str:
        if features is None or (hasattr(features, "__len__") and len(features) == 0):
            return ""
        if isinstance(features, list):
            return " ".join(str(f) for f in features)
        return str(features)

    item_features["description"] = item_features["feature"].apply(extract_description)
    item_features = item_features.drop_duplicates(subset=["asin"])

    logger.info("Processed %d unique items.", len(item_features))
    return item_features


def process_user_features(
    review_df: pd.DataFrame, min_purchases: int = DEFAULT_MIN_PURCHASES
) -> pd.DataFrame:
    """Build user purchase histories from review data.

    Args:
        review_df: Raw review DataFrame.
        min_purchases: Minimum number of verified purchases to keep a user.

    Returns:
        DataFrame with columns: reviewerID, asin (list of products in order).
    """
    logger.info("Processing user features...")
    verified_reviews = review_df[review_df["verified"] == True].copy()

    user_purchase_counts = verified_reviews.groupby("reviewerID").size()
    active_users = user_purchase_counts[user_purchase_counts >= min_purchases].index

    filtered_reviews = verified_reviews[
        verified_reviews["reviewerID"].isin(active_users)
    ].copy()

    filtered_reviews["timestamp"] = pd.to_datetime(
        filtered_reviews["unixReviewTime"], unit="s"
    )
    filtered_reviews = filtered_reviews.sort_values(["reviewerID", "timestamp"])

    user_purchase_history = (
        filtered_reviews.groupby("reviewerID")
        .agg({"asin": list, "timestamp": list})
        .reset_index()
    )

    logger.info(
        "Processed %d users (min %d purchases), %d total records.",
        len(user_purchase_history),
        min_purchases,
        len(filtered_reviews),
    )
    return user_purchase_history


# ── Encoding ─────────────────────────────────────────────


def create_label_encoders(
    item_features: pd.DataFrame,
) -> tuple[LabelEncoder, LabelEncoder]:
    """Fit label encoders for products and brands.

    Args:
        item_features: Cleaned item features DataFrame.

    Returns:
        Tuple of (product_encoder, brand_encoder).
    """
    logger.info("Creating label encoders...")

    product_encoder = LabelEncoder()
    product_encoder.fit(item_features["asin"])

    brand_encoder = LabelEncoder()
    brand_encoder.fit(item_features["brand"])

    logger.info("Product encoder: %d unique products.", len(product_encoder.classes_))
    logger.info("Brand encoder: %d unique brands.", len(brand_encoder.classes_))

    return product_encoder, brand_encoder


def encode_item_features(
    item_features: pd.DataFrame,
    product_encoder: LabelEncoder,
    brand_encoder: LabelEncoder,
) -> pd.DataFrame:
    """Apply label encoders to item features.

    Args:
        item_features: Cleaned item features DataFrame.
        product_encoder: Fitted product LabelEncoder.
        brand_encoder: Fitted brand LabelEncoder.

    Returns:
        DataFrame with columns: product_id, title, description, brand_id.
    """
    logger.info("Encoding item features...")
    encoded = item_features.copy()
    encoded["product_id"] = product_encoder.transform(encoded["asin"])
    encoded["brand_id"] = brand_encoder.transform(encoded["brand"])

    result = encoded[["product_id", "title", "description", "brand_id"]]
    logger.info("Final item features shape: %s", result.shape)
    return result


def encode_user_features(
    user_features: pd.DataFrame,
    product_encoder: LabelEncoder,
    max_history_length: int = DEFAULT_MAX_HISTORY_LENGTH,
) -> pd.DataFrame:
    """Encode user histories as padded sequences of product indices.

    Args:
        user_features: User purchase history DataFrame.
        product_encoder: Fitted product LabelEncoder.
        max_history_length: Maximum sequence length (truncated/padded).

    Returns:
        DataFrame with columns: reviewerID, history_padded.
    """
    logger.info("Encoding user features...")
    product_to_idx = {
        p: idx for idx, p in enumerate(product_encoder.classes_)
    }

    def encode_and_pad(product_list: list[str]) -> list[int]:
        encoded = [product_to_idx[p] for p in product_list if p in product_to_idx]
        encoded = encoded[-max_history_length:]
        pad_len = max_history_length - len(encoded)
        if pad_len > 0:
            encoded.extend([0] * pad_len)
        return encoded

    encoded_users = user_features[["reviewerID", "asin"]].copy()
    encoded_users["history_padded"] = encoded_users["asin"].apply(encode_and_pad)
    result = encoded_users[["reviewerID", "history_padded"]]

    logger.info("Final user features shape: %s", result.shape)
    return result


# ── Training Pair Generation ─────────────────────────────


def create_training_pairs(
    user_features: pd.DataFrame,
    max_context_length: int = 50,
) -> pd.DataFrame:
    """Generate (user, target_item, context_history) training triplets.

    For each user with >= 3 interactions, creates one training sample per
    item after the second, using all preceding items as context.

    Args:
        user_features: Encoded user features with history_padded column.
        max_context_length: Maximum length to pad the context history.

    Returns:
        DataFrame with columns: user_id, target_item_id, context_history.
    """
    logger.info("Creating training pairs...")
    training_pairs: list[dict[str, Any]] = []

    for _, user_row in user_features.iterrows():
        user_id = user_row["reviewerID"]
        history = user_row["history_padded"]

        # Remove padding (0s)
        actual_history = [int(p) for p in history if p != 0]

        if len(actual_history) < 3:
            continue

        for i in range(2, len(actual_history)):
            target_item = actual_history[i]
            context_history = list(actual_history[:i])

            # Pad context to fixed length
            if len(context_history) < max_context_length:
                context_history += [0] * (max_context_length - len(context_history))

            training_pairs.append({
                "user_id": user_id,
                "target_item_id": target_item,
                "context_history": context_history,
            })

    training_df = pd.DataFrame(training_pairs)
    logger.info("Created %d training pairs.", len(training_df))
    return training_df


# ── Save ─────────────────────────────────────────────────


def save_data(
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    training_pairs: pd.DataFrame,
    product_encoder: LabelEncoder,
    brand_encoder: LabelEncoder,
    output_dir: str | Path,
) -> None:
    """Save processed data and fitted encoders to disk.

    Args:
        item_features: Encoded item features.
        user_features: Encoded user features.
        training_pairs: Training triplets.
        product_encoder: Fitted product encoder.
        brand_encoder: Fitted brand encoder.
        output_dir: Directory to save files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    item_features.to_parquet(output_path / "item_features.parquet", index=False)
    user_features.to_parquet(output_path / "user_features.parquet", index=False)
    training_pairs.to_parquet(output_path / "train_pairs.parquet", index=False)

    with open(output_path / "product_encoder.pkl", "wb") as f:
        pickle.dump(product_encoder, f)
    with open(output_path / "brand_encoder.pkl", "wb") as f:
        pickle.dump(brand_encoder, f)

    logger.info("Data saved to %s.", output_path)


# ── Main Pipeline ────────────────────────────────────────


def main() -> None:
    """Run the full data processing pipeline."""
    # File paths
    review_file = Path(__file__).parent / "dataset" / "AMAZON_FASHION.json.gz"
    meta_file = Path(__file__).parent / "dataset" / "meta_AMAZON_FASHION.json.gz"
    output_dir = Path(__file__).parent / "processed_data"

    # Load data
    review_df = load_jsonl_gz(review_file)
    product_df = load_jsonl_gz(meta_file)

    # Process features
    item_features = process_item_features(product_df)
    user_features = process_user_features(review_df, min_purchases=3)

    # Create encoders
    product_encoder, brand_encoder = create_label_encoders(item_features)

    # Encode features
    encoded_items = encode_item_features(item_features, product_encoder, brand_encoder)
    encoded_users = encode_user_features(user_features, product_encoder)

    # Create training pairs
    training_pairs = create_training_pairs(encoded_users)

    # Save all data
    save_data(encoded_items, encoded_users, training_pairs,
              product_encoder, brand_encoder, output_dir)

    # Print summary
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Items processed: {len(encoded_items)}")
    print(f"Users processed: {len(encoded_users)}")
    print(f"Training pairs created: {len(training_pairs)}")
    print(f"Unique products: {len(product_encoder.classes_)}")
    print(f"Unique brands: {len(brand_encoder.classes_)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
