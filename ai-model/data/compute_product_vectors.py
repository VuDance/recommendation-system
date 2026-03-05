import pandas as pd
import pickle
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from main.product_tower import ProductTower
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_product_tower(checkpoint_path):
    """Load pre-trained ProductTower from checkpoint"""
    logger.info(f"Loading ProductTower from {checkpoint_path}")
    
    # Create model with same architecture
    product_tower = ProductTower(brand_vocab_size=18513)  # From our encoder
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    product_tower.load_state_dict(checkpoint['model_state_dict'])
    product_tower.eval()
    
    logger.info("ProductTower loaded successfully")
    return product_tower

def compute_text_embeddings(texts, batch_size=128):
    """Compute text embeddings using sentence-transformers"""
    logger.info(f"Computing text embeddings for {len(texts)} products")
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
        
        if i % (batch_size * 10) == 0:
            logger.info(f"Processed {i}/{len(texts)} products")
    
    embeddings = torch.cat(embeddings, dim=0)
    logger.info(f"Text embeddings shape: {embeddings.shape}")
    return embeddings

def compute_product_vectors():
    """Main function to compute and store product vectors"""
    # Load data
    logger.info("Loading item features...")
    item_df = pd.read_parquet('processed_data/item_features.parquet')
    logger.info(f"Loaded {len(item_df)} products")
    
    # Load ProductTower
    product_tower = load_product_tower('model/product_tower_checkpoint.pth')
    
    # Prepare data
    batch_size = 128
    all_vectors = []
    all_product_ids = []
    all_product_idxs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    product_tower = product_tower.to(device)
    
    logger.info("Computing product vectors...")
    
    for i in range(0, len(item_df), batch_size):
        batch_df = item_df.iloc[i:i+batch_size]
        
        # Prepare text inputs
        titles = batch_df['title'].tolist()
        descriptions = batch_df['description'].tolist()
        text_inputs = [f"{title} {desc}" for title, desc in zip(titles, descriptions)]
        
        # Compute text embeddings
        text_embeddings = compute_text_embeddings(text_inputs, batch_size=len(text_inputs))
        
        # Prepare other features
        brand_ids = torch.tensor(batch_df['brand_id'].values, dtype=torch.long)
        prices = torch.tensor(batch_df['price_normalized'].values, dtype=torch.float32)
        
        # Compute product vectors
        with torch.no_grad():
            product_vectors = product_tower(
                text_embeddings.to(device),
                brand_ids.to(device),
                prices.to(device)
            )
        
        # Store results
        all_vectors.append(product_vectors.cpu().numpy())
        all_product_ids.extend(batch_df['product_id'].tolist())
        all_product_idxs.extend(batch_df.index.tolist())
        
        if i % (batch_size * 10) == 0:
            logger.info(f"Processed {i}/{len(item_df)} products")
    
    # Combine all vectors
    all_vectors = np.vstack(all_vectors)
    logger.info(f"Final vectors shape: {all_vectors.shape}")
    logger.info(f"Sample vector: {all_vectors[0][:5]}...")
    
    # Save vectors
    output_dir = Path('processed_data')
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'product_vectors.npy', all_vectors)
    
    # Save metadata
    metadata = {
        'product_ids': all_product_ids,
        'product_idxs': all_product_idxs,
        'vectors_shape': all_vectors.shape
    }
    
    with open(output_dir / 'product_vectors_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info("Product vectors saved successfully!")
    logger.info(f"Vectors shape: {all_vectors.shape}")
    logger.info(f"Product IDs: {len(all_product_ids)}")
    logger.info(f"Product indices: {len(all_product_idxs)}")
    
    return all_vectors, all_product_ids, all_product_idxs

def main():
    """Main execution"""
    try:
        vectors, product_ids, product_idxs = compute_product_vectors()
        
        print("\n=== PRODUCT VECTOR COMPUTATION COMPLETE ===")
        print(f"Vectors shape: {vectors.shape}")
        print(f"Sample vector (first 5 dims): {vectors[0][:5]}")
        print(f"Product IDs sample: {product_ids[:5]}")
        print(f"Product indices sample: {product_idxs[:5]}")
        
    except Exception as e:
        logger.error(f"Error in compute_product_vectors: {e}")
        raise

if __name__ == "__main__":
    main()