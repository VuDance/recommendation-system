#!/usr/bin/env python3
"""
Store product vectors in Milvus vector database
"""
import numpy as np
import pandas as pd
import pickle
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_milvus():
    """Connect to Milvus database"""
    logger.info("Connecting to Milvus...")
    try:
        connections.connect(host="localhost", port="19530")
        logger.info("Connected to Milvus successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return False

def create_collection():
    """Create Milvus collection with proper schema"""
    logger.info("Creating Milvus collection...")
    
    # Define fields
    fields = [
        FieldSchema(name="product_idx", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=64)
    ]
    
    # Create schema
    schema = CollectionSchema(fields, description="Product vectors for recommendation")
    
    # Create collection
    collection_name = "product_vectors"
    collection = Collection(name=collection_name, schema=schema)
    
    logger.info(f"Collection '{collection_name}' created successfully")
    return collection

def create_index(collection):
    """Create IVF_FLAT index on vector field"""
    logger.info("Creating IVF_FLAT index...")
    
    index_params = {
        "metric_type": "IP",  # Inner Product
        "params": {"nlist": 100}  # Number of clusters
    }
    
    collection.create_index(field_name="vector", index_params=index_params)
    logger.info("Index created successfully")

def load_product_data():
    """Load precomputed product vectors and metadata"""
    logger.info("Loading product vectors and metadata...")
    
    # Load vectors
    vectors = np.load('processed_data/product_vectors.npy')
    
    # Load metadata
    with open('processed_data/product_vectors_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded vectors shape: {vectors.shape}")
    logger.info(f"Product IDs: {len(metadata['product_ids'])}")
    logger.info(f"Product indices: {len(metadata['product_idxs'])}")
    
    return vectors, metadata

def insert_batch(collection, batch_data, batch_size=1000):
    """Insert data in batches to Milvus"""
    logger.info(f"Inserting {len(batch_data['product_idx'])} records in batches of {batch_size}")
    
    total_records = len(batch_data['product_idx'])
    
    for i in range(0, total_records, batch_size):
        batch_end = min(i + batch_size, total_records)
        
        batch = {
            'product_idx': batch_data['product_idx'][i:batch_end],
            'product_id': batch_data['product_id'][i:batch_end],
            'vector': batch_data['vector'][i:batch_end]
        }
        
        # Insert batch
        collection.insert([
            batch['product_idx'],
            batch['product_id'],
            batch['vector']
        ])
        
        if i % (batch_size * 10) == 0:
            logger.info(f"Inserted {batch_end}/{total_records} records")
    
    # Flush to make data persistent
    collection.flush()
    logger.info("Data flushed to Milvus")

def store_in_milvus():
    """Main function to store product vectors in Milvus"""
    # Connect to Milvus
    if not connect_to_milvus():
        return False
    
    # Load data
    vectors, metadata = load_product_data()
    
    # Create collection
    collection = create_collection()
    
    # Create index
    create_index(collection)
    
    # Prepare batch data
    batch_data = {
        'product_idx': metadata['product_idxs'],
        'product_id': [str(pid) for pid in metadata['product_ids']],  # Convert to string
        'vector': vectors.tolist()  # Convert to list for Milvus
    }
    
    # Insert data
    insert_batch(collection, batch_data)
    
    # Verify insertion
    logger.info(f"Collection row count: {collection.num_entities}")
    
    return True

def main():
    """Main execution"""
    try:
        success = store_in_milvus()
        
        if success:
            print("\n=== MILVUS STORAGE COMPLETE ===")
            print("Product vectors successfully stored in Milvus")
            print("Collection: product_vectors")
            print("Fields: product_idx (INT64), product_id (VARCHAR), vector (FLOAT_VECTOR)")
            print("Index: IVF_FLAT with IP metric")
        else:
            print("\n=== MILVUS STORAGE FAILED ===")
            print("Failed to store vectors in Milvus")
        
    except Exception as e:
        logger.error(f"Error in store_in_milvus: {e}")
        raise

if __name__ == "__main__":
    main()