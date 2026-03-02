import pandas as pd
import pickle
import torch
from main.product_tower import ProductTower
import numpy as np
from pathlib import Path
import logging
from main.util.util import connect_to_milvus, load_encoders
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_product_vectors():
    logger.info("=== COMPUTING PRODUCT VECTORS ===")

    item_df = pd.read_parquet('processed_data/item_features.parquet')
    item_df = item_df.reset_index(drop=True)  # ✅ FIX: đảm bảo index liên tục từ 0

    brand_encoder = load_encoders()

    # ✅ FIX: Load text model MỘT LẦN duy nhất, ngoài vòng lặp
    logger.info("Loading SentenceTransformer (once)...")
    try:
        from sentence_transformers import SentenceTransformer
        text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except ImportError:
        logger.warning("sentence-transformers not available, using random embeddings")
        text_model = None

    product_tower = ProductTower(brand_vocab_size=len(brand_encoder.classes_))
    product_tower.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    product_tower = product_tower.to(device)

    batch_size = 128
    all_vectors, all_product_ids, all_product_idxs = [], [], []

    for i in range(0, len(item_df), batch_size):
        batch_df    = item_df.iloc[i:i + batch_size]
        text_inputs = [f"{r['title']} {r['description']}" for _, r in batch_df.iterrows()]

        # ✅ FIX: dùng text_model đã load sẵn
        if text_model is not None:
            text_embeddings = text_model.encode(text_inputs, convert_to_tensor=True, show_progress_bar=False)
        else:
            text_embeddings = torch.randn(len(batch_df), 384)

        brand_ids = torch.tensor(batch_df['brand_id'].values, dtype=torch.long)
        prices    = torch.zeros(len(batch_df), dtype=torch.float32)

        with torch.no_grad():
            product_vectors = product_tower(
                text_embeddings.to(device),
                brand_ids.to(device),
                prices.to(device)
            )

        all_vectors.append(product_vectors.cpu().numpy())
        all_product_ids.extend(batch_df['product_id'].tolist())
        # ✅ FIX: dùng range liên tục thay vì batch_df.index (tránh gap)
        all_product_idxs.extend(list(range(i, i + len(batch_df))))

        if i % (batch_size * 10) == 0:
            logger.info(f"Processed {i + len(batch_df)}/{len(item_df)}")

    all_vectors = np.vstack(all_vectors)

    output_dir = Path('processed_data')
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / 'product_vectors.npy', all_vectors)

    metadata = {
        'product_ids'   : all_product_ids,
        'product_idxs'  : all_product_idxs,
        'vectors_shape' : all_vectors.shape
    }
    with open(output_dir / 'product_vectors_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    logger.info(f"Saved vectors shape: {all_vectors.shape}")
    return all_vectors, metadata


def create_milvus_collection():
    try:
        collection_name = "product_vectors"
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(name="product_idx", dtype=DataType.INT64,         is_primary=True, auto_id=False),
            FieldSchema(name="product_id",  dtype=DataType.VARCHAR,       max_length=100),
            FieldSchema(name="vector",      dtype=DataType.FLOAT_VECTOR,  dim=64)
        ]
        schema     = CollectionSchema(fields, description="Product vectors for recommendation")
        collection = Collection(name=collection_name, schema=schema)
        logger.info(f"Collection '{collection_name}' created!")
        return collection
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return None


def insert_vectors_to_milvus(collection, vectors, metadata, batch_size=1000):
    total = len(metadata['product_ids'])
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        try:
            collection.insert([
                metadata['product_idxs'][i:end],
                [str(pid) for pid in metadata['product_ids'][i:end]],
                vectors[i:end].tolist()
            ])
        except Exception as e:
            logger.error(f"Insert batch {i // batch_size} failed: {e}")

    collection.flush()
    logger.info("Data flushed!")


def create_milvus_index(collection):
    # ✅ FIX: gọi SAU khi đã insert xong
    try:
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type" : "IVF_FLAT",
                "metric_type": "IP",        # Inner Product vì đã L2 normalize
                "params"     : {"nlist": 128}
            }
        )
        logger.info("Index created!")
    except Exception as e:
        logger.error(f"Index failed: {e}")


def validate_milvus(collection):
    try:
        # ✅ FIX: phải load vào memory trước khi search
        collection.load()
        count = collection.num_entities
        logger.info(f"Collection loaded. Row count: {count}")
        return count > 0
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    print("🚀 MILVUS INTEGRATION")
    print("=" * 50)

    print("\n1. Computing product vectors...")
    vectors, metadata = compute_product_vectors()

    print("\n2. Connecting to Milvus...")
    if not connect_to_milvus():
        return

    print("\n3. Creating collection...")
    collection = create_milvus_collection()
    if collection is None:
        return

    # ✅ Đúng thứ tự: Insert → Flush → Index → Load
    print("\n4. Inserting vectors...")
    insert_vectors_to_milvus(collection, vectors, metadata)

    print("\n5. Creating index (after insert)...")
    create_milvus_index(collection)

    print("\n6. Loading + validating...")
    success = validate_milvus(collection)

    print(f"\n=== RESULTS ===")
    print(f"Vectors : {vectors.shape}")
    print(f"Index   : IVF_FLAT | IP metric")
    print(f"Status  : {'✅ SUCCESS' if success else '❌ FAILED'}")


if __name__ == "__main__":
    main()