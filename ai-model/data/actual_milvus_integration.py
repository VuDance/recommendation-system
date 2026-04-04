"""Milvus vector database integration for product embeddings.

Handles computing product vectors using the ProductTower model
and inserting them into a Milvus collection for similarity search.
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.product_tower import ProductTower
from main.util.util import connect_to_milvus, load_encoders
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────

COLLECTION_NAME: str = "product_vectors"
VECTOR_DIM: int = 64
BATCH_SIZE: int = 128
INSERT_BATCH_SIZE: int = 1000
IVF_NLIST: int = 128
PRODUCT_ID_MAX_LENGTH: int = 100


# ── Product Vector Computation ───────────────────────────


def compute_product_vectors(
    data_dir: str = "processed_data",
    model_checkpoint: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute product embeddings using the trained ProductTower model.

    Args:
        data_dir: Directory containing processed data and encoders.
        model_checkpoint: Path to a ProductTower checkpoint. If None, uses
                          an untrained model.
        batch_size: Number of items to process per batch.

    Returns:
        Tuple of (vectors_array, metadata_dict).
    """
    logger.info("=== COMPUTING PRODUCT VECTORS ===")

    data_path = Path(data_dir)
    item_df = pd.read_parquet(data_path / "item_features.parquet")
    # Ensure continuous index from 0
    item_df = item_df.reset_index(drop=True)

    _, brand_encoder = load_encoders(str(data_path))

    # Load SentenceTransformer once
    logger.info("Loading SentenceTransformer...")
    try:
        from sentence_transformers import SentenceTransformer
        text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        logger.warning("sentence-transformers not available, using random embeddings")
        text_model = None

    # Initialize ProductTower
    product_tower = ProductTower(
        brand_vocab_size=len(brand_encoder.classes_),
        output_dim=VECTOR_DIM,
    )

    # Load checkpoint if provided
    if model_checkpoint:
        logger.info("Loading model checkpoint from %s", model_checkpoint)
        checkpoint = torch.load(model_checkpoint, map_location="cpu", weights_only=True)
        product_tower.load_state_dict(checkpoint["model_state_dict"])

    product_tower.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    product_tower = product_tower.to(device)

    all_vectors: list[np.ndarray] = []
    all_product_ids: list[str] = []
    all_product_idxs: list[int] = []

    for i in range(0, len(item_df), batch_size):
        batch_df = item_df.iloc[i : i + batch_size]
        text_inputs = [
            f"{row['title']} {row['description']}"
            for _, row in batch_df.iterrows()
        ]

        # Encode text
        if text_model is not None:
            text_embeddings = text_model.encode(
                text_inputs, convert_to_tensor=True, show_progress_bar=False
            )
        else:
            text_embeddings = torch.randn(len(batch_df), 384)

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
        # Use continuous range to avoid index gaps
        all_product_idxs.extend(range(i, i + len(batch_df)))

        if (i // batch_size + 1) % 10 == 0:
            logger.info("Processed %d/%d items", i + len(batch_df), len(item_df))

    all_vectors_array = np.vstack(all_vectors)

    # Save vectors and metadata
    np.save(data_path / "product_vectors.npy", all_vectors_array)

    metadata = {
        "product_ids": all_product_ids,
        "product_idxs": all_product_idxs,
        "vectors_shape": all_vectors_array.shape,
    }
    with open(data_path / "product_vectors_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    logger.info("Saved vectors shape: %s", all_vectors_array.shape)
    return all_vectors_array, metadata


# ── Milvus Operations ────────────────────────────────────


def create_milvus_collection(
    collection_name: str = COLLECTION_NAME,
    vector_dim: int = VECTOR_DIM,
) -> Collection | None:
    """Create a new Milvus collection, dropping any existing one with the same name.

    Args:
        collection_name: Name of the collection to create.
        vector_dim: Dimension of the product embedding vectors.

    Returns:
        The created Collection, or None if creation failed.
    """
    try:
        if utility.has_collection(collection_name):
            logger.info("Dropping existing collection '%s'", collection_name)
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(
                name="product_idx",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="product_id",
                dtype=DataType.VARCHAR,
                max_length=PRODUCT_ID_MAX_LENGTH,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=vector_dim,
            ),
        ]
        schema = CollectionSchema(
            fields, description="Product vectors for recommendation"
        )
        collection = Collection(name=collection_name, schema=schema)
        logger.info("Collection '%s' created.", collection_name)
        return collection
    except Exception as exc:
        logger.error("Failed to create collection: %s", exc)
        return None


def insert_vectors_to_milvus(
    collection: Collection,
    vectors: np.ndarray,
    metadata: dict[str, Any],
    batch_size: int = INSERT_BATCH_SIZE,
) -> None:
    """Insert product vectors and metadata into Milvus in batches.

    Args:
        collection: Target Milvus collection.
        vectors: Array of product vectors of shape (num_products, dim).
        metadata: Dict containing product_ids and product_idxs lists.
        batch_size: Number of records per insert batch.
    """
    total = len(metadata["product_ids"])
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        try:
            collection.insert([
                metadata["product_idxs"][i:end],
                [str(pid) for pid in metadata["product_ids"][i:end]],
                vectors[i:end].tolist(),
            ])
        except Exception as exc:
            logger.error("Insert batch %d failed: %s", i // batch_size, exc)

    collection.flush()
    logger.info("Data flushed! Total inserted: %s", total)


def create_milvus_index(
    collection: Collection,
    index_type: str = "IVF_FLAT",
    metric_type: str = "IP",
    nlist: int = IVF_NLIST,
) -> None:
    """Create an IVF_FLAT index on the vector field.

    Args:
        collection: Target Milvus collection.
        index_type: Type of index to create.
        metric_type: Distance metric (IP = Inner Product).
        nlist: Number of cluster centers for IVF.
    """
    try:
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": index_type,
                "metric_type": metric_type,
                "params": {"nlist": nlist},
            },
        )
        logger.info("Index created!")
    except Exception as exc:
        logger.error("Index creation failed: %s", exc)


def validate_milvus(collection: Collection) -> bool:
    """Load the collection and verify it has data.

    Args:
        collection: Target Milvus collection.

    Returns:
        True if the collection has at least one entity.
    """
    try:
        collection.load()
        count = collection.num_entities
        logger.info("Collection loaded. Row count: %s", count)
        return count > 0
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        return False


# ── Main Pipeline ────────────────────────────────────────


def main() -> None:
    """Run the full Milvus integration pipeline."""
    # Resolve data path relative to this script's location
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / "processed_data"

    print("MILVUS INTEGRATION")
    print("=" * 50)

    print("\n1. Computing product vectors...")
    vectors, metadata = compute_product_vectors(data_dir=str(data_dir))

    print("\n2. Connecting to Milvus...")
    if not connect_to_milvus():
        return

    print("\n3. Creating collection...")
    collection = create_milvus_collection()
    if collection is None:
        return

    print("\n4. Inserting vectors...")
    insert_vectors_to_milvus(collection, vectors, metadata)

    print("\n5. Creating index (after insert)...")
    create_milvus_index(collection)

    print("\n6. Loading + validating...")
    success = validate_milvus(collection)

    print(f"\n=== RESULTS ===")
    print(f"Vectors : {vectors.shape}")
    print(f"Index   : IVF_FLAT | IP metric")
    print(f"Status  : {'SUCCESS' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
