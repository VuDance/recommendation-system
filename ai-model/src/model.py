"""Content-based similarity and recommendation model using Milvus ANN search.

Product metadata is encoded into dense 384-dim vectors via
SentenceTransformer (all-MiniLM-L6-v2).  Vectors are L2-normalised and
stored in Milvus with the Inner Product metric (equivalent to cosine
similarity on normalised vectors).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "content_based_products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MILVUS_USER = "root"
MILVUS_PASSWORD = "Milvus"


# ── Embedding helper ─────────────────────────────────────────────────


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode text strings via SentenceTransformer → L2-normalised vectors.

    Args:
        texts: List of product text strings (title + description + brand).

    Returns:
        L2-normalised dense vectors of shape (n_texts, 384).
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    return vectors / norms


def encode_single(text: str) -> np.ndarray:
    """Encode a single text query.

    Args:
        text: Query text.

    Returns:
        L2-normalised vector of shape (384,).
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector = model.encode(text, convert_to_numpy=True)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


# ── Milvus collection management ─────────────────────────────────────


def init_milvus_collection(dim: int = EMBEDDING_DIM) -> None:
    """Create the product embedding collection if it does not exist.

    Args:
        dim: Vector dimensionality (default 384 from SentenceTransformer).
    """
    client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)

    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)
        logger.info("Dropped existing collection '%s'.", COLLECTION_NAME)

    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="product_idx", datatype=DataType.INT64)
    schema.add_field(field_name="product_id", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="brand", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name="vector_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )
    logger.info("Collection '%s' created, dim=%d", COLLECTION_NAME, dim)


def ingest_vectors(
    vectors: np.ndarray,
    product_ids: list[str],
    product_idxs: list[int],
    brands: list[str] | None = None,
    batch_size: int = 5000,
) -> None:
    """Insert product embeddings into Milvus in batches.

    Args:
        vectors: Dense embeddings of shape (n_items, dim).
        product_ids: List of product IDs per row.
        product_idxs: Integer indices (row positions in train corpus).
        brands: Optional brand names per row.
        batch_size: Number of vectors per insert call.
    """
    client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)

    if brands is None:
        brands = [""] * len(vectors)

    total = len(vectors)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch = [
            {
                "product_idx": int(product_idxs[j]),
                "product_id": str(product_ids[j]),
                "brand": str(brands[j]),
                "vector": vectors[j].tolist(),
            }
            for j in range(i, end)
        ]
        client.insert(collection_name=COLLECTION_NAME, data=batch)
        logger.info("Inserted batch %d/%d (%d vectors)", i + batch_size, total, len(batch))

    logger.info("Ingested %d vectors total into Milvus.", total)


# ── Retrieval via Milvus ANN ─────────────────────────────────────────


def search_similar(
    query_vector: np.ndarray,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search Milvus for the top_k most similar products.

    Args:
        query_vector: L2-normalised query vector (dim,).
        top_k: Number of recommendations.

    Returns:
        List of dicts with keys: product_id, product_idx, brand, score, rank.
    """
    client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)
    client.load_collection(collection_name=COLLECTION_NAME)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector.tolist()],
        limit=top_k,
        output_fields=["product_idx", "product_id", "brand"],
        search_params={"metric_type": "IP", "params": {"nprobe": 16}},
    )

    return [
        {
            "product_id": hit["entity"]["product_id"],
            "product_idx": int(hit["entity"]["product_idx"]),
            "brand": hit["entity"]["brand"],
            "score": hit["distance"],
            "rank": rank + 1,
        }
        for rank, hit in enumerate(results[0])
    ]


def search_similar_batch(
    query_vectors: np.ndarray,
    top_k: int = 50,
) -> list[list[dict[str, Any]]]:
    """Batch-search Milvus for multiple query vectors.

    Args:
        query_vectors: L2-normalised query vectors (n_queries, dim).
        top_k: Number of recommendations per query.

    Returns:
        List per query of recommendation dicts.
    """
    client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)
    client.load_collection(collection_name=COLLECTION_NAME)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vectors.tolist(),
        limit=top_k,
        output_fields=["product_idx", "product_id", "brand"],
        search_params={"metric_type": "IP", "params": {"nprobe": 16}},
    )

    batch_recs = []
    for hits in results:
        batch_recs.append(
            [
                {
                    "product_id": hit["entity"]["product_id"],
                    "product_idx": int(hit["entity"]["product_idx"]),
                    "brand": hit["entity"]["brand"],
                    "score": hit["distance"],
                    "rank": rank + 1,
                }
                for rank, hit in enumerate(hits)
            ]
        )
    return batch_recs


# ── High-level recommendation endpoint ───────────────────────────────


def get_recommendations(
    query_text: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Generate content-based recommendations for a single text query.

    Args:
        query_text: Input text (e.g. a product title + brand + description).
        top_k: Number of recommendations.

    Returns:
        List of recommendation dicts.
    """
    query_vector = encode_single(query_text)
    return search_similar(query_vector, top_k=top_k)
