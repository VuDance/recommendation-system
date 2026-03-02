#!/usr/bin/env python3
"""
Data processing script for Two-Tower recommendation model
Processes Amazon Fashion dataset into training data format
"""
import pandas as pd
import gzip
import json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_review_data(filepath, max_rows=None):
    """Load review data from gzipped JSONL file"""
    logger.info(f"Loading review data from {filepath}")
    
    samples = []
    with gzip.open(filepath, 'rt') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            try:
                data = json.loads(line)
                samples.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {i}: {e}")
                continue
    
    logger.info(f"Loaded {len(samples)} review samples")
    return pd.DataFrame(samples)

def load_product_metadata(filepath, max_rows=None):
    """Load product metadata from gzipped JSONL file"""
    logger.info(f"Loading product metadata from {filepath}")
    
    samples = []
    with gzip.open(filepath, 'rt') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            try:
                data = json.loads(line)
                samples.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {i}: {e}")
                continue
    
    logger.info(f"Loaded {len(samples)} product metadata samples")
    return pd.DataFrame(samples)

def process_item_features(product_df):
    """Process item features from product metadata"""
    logger.info("Processing item features...")
    
    # Select relevant columns and handle missing values
    item_features = product_df[['asin', 'title', 'brand', 'feature']].copy()
    
    # Handle missing values
    item_features['title'] = item_features['title'].fillna('')
    item_features['brand'] = item_features['brand'].fillna('Unknown')
    item_features['feature'] = item_features['feature'].fillna('')
    
    # Extract description from feature if available
    def extract_description(features):
        if features is None or (hasattr(features, '__len__') and len(features) == 0):
            return ''
        if isinstance(features, list):
            return ' '.join(str(f) for f in features)
        return str(features)
    
    item_features['description'] = item_features['feature'].apply(extract_description)
    
    # Remove duplicates (keep first occurrence)
    item_features = item_features.drop_duplicates(subset=['asin'])
    
    logger.info(f"Processed {len(item_features)} unique items")
    return item_features

def process_user_features(review_df, min_purchases=2):
    """Process user features from review data"""
    logger.info("Processing user features...")
    
    # Filter reviews to only include those with verified purchases
    verified_reviews = review_df[review_df['verified'] == True].copy()
    
    # Count purchases per user
    user_purchase_counts = verified_reviews.groupby('reviewerID').size()
    active_users = user_purchase_counts[user_purchase_counts >= min_purchases].index
    
    # Filter to active users only
    filtered_reviews = verified_reviews[verified_reviews['reviewerID'].isin(active_users)].copy()
    
    # Sort by user and timestamp
    filtered_reviews['timestamp'] = pd.to_datetime(filtered_reviews['unixReviewTime'], unit='s')
    filtered_reviews = filtered_reviews.sort_values(['reviewerID', 'timestamp'])
    
    # Group by user to get purchase history
    user_purchase_history = filtered_reviews.groupby('reviewerID').agg({
        'asin': list,
        'timestamp': list
    }).reset_index()
    
    logger.info(f"Processed {len(user_purchase_history)} users with at least {min_purchases} purchases")
    logger.info(f"Total purchase records: {len(filtered_reviews)}")
    
    return user_purchase_history

def create_label_encoders(item_features, user_features):
    """Create and fit label encoders"""
    logger.info("Creating label encoders...")
    
    # Product encoder
    product_encoder = LabelEncoder()
    product_encoder.fit(item_features['asin'])
    
    # Brand encoder
    brand_encoder = LabelEncoder()
    brand_encoder.fit(item_features['brand'])
    
    # # User encoder
    # user_encoder = LabelEncoder()
    # user_encoder.fit(user_features['reviewerID'])
    
    logger.info(f"Product encoder: {len(product_encoder.classes_)} unique products")
    logger.info(f"Brand encoder: {len(brand_encoder.classes_)} unique brands")
    # logger.info(f"User encoder: {len(user_encoder.classes_)} unique users")
    
    return product_encoder, brand_encoder

def encode_item_features(item_features, product_encoder, brand_encoder):
    """Encode item features using label encoders"""
    logger.info("Encoding item features...")
    
    encoded_items = item_features.copy()
    
    # Encode product_id to sequential integers
    encoded_items['product_id'] = product_encoder.transform(encoded_items['asin'])
    
    # Encode brand to sequential integers
    encoded_items['brand_id'] = brand_encoder.transform(encoded_items['brand'])
    
    # Select final columns
    final_columns = ['product_id', 'title', 'description', 'brand_id']
    encoded_items = encoded_items[final_columns]
    
    logger.info(f"Final item features shape: {encoded_items.shape}")
    return encoded_items

def encode_user_features(user_features, product_encoder, max_history_length=50):
    logger.info("Encoding user features...")
    encoded_users = user_features[['reviewerID', 'asin']].copy()  # ← đổi user_id → reviewerID

    product_to_idx = dict(zip(
        product_encoder.classes_, 
        range(len(product_encoder.classes_))
    ))

    def encode_and_pad(product_list):
        encoded = [product_to_idx[p] for p in product_list if p in product_to_idx]
        encoded = encoded[-max_history_length:]
        pad_len = max_history_length - len(encoded)
        if pad_len > 0:
            encoded.extend([0] * pad_len)
        return encoded

    encoded_users['history_padded'] = encoded_users['asin'].apply(encode_and_pad)
    encoded_users = encoded_users[['reviewerID', 'history_padded']]  # ← đổi user_id → reviewerID

    logger.info(f"Final user features shape: {encoded_users.shape}")
    return encoded_users

def create_training_pairs(user_features, product_encoder):
    """Create training pairs (user, target_item, history)"""
    logger.info("Creating training pairs...")
    
    training_pairs = []
    
    for _, user_row in user_features.iterrows():
        user_id = user_row['reviewerID']
        history = user_row['history_padded']
        
        # Remove padding (0s) from history
        actual_history = [p for p in history if p != 0]
        
        if len(actual_history) < 3:  # Should not happen due to filtering, but safety check
            continue
            
        # Create pairs for each item in history (except the first few)
        for i in range(2, len(actual_history)):
            target_item = actual_history[i]
            context_history = actual_history[:i]  # History before target
            
            # Pad context history if needed
            if len(context_history) < 50:
                context_history = context_history + [0] * (50 - len(context_history))
            
            training_pairs.append({
                'user_id': user_id,
                'target_item_id': target_item,
                'context_history': context_history
            })
    
    training_df = pd.DataFrame(training_pairs)
    logger.info(f"Created {len(training_df)} training pairs")
    return training_df

def save_data(item_features, user_features, training_pairs, 
              product_encoder, brand_encoder, output_dir):
    """Save all processed data and encoders"""
    logger.info(f"Saving data to {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save parquet files
    item_features.to_parquet(f"{output_dir}/item_features.parquet", index=False)
    user_features.to_parquet(f"{output_dir}/user_features.parquet", index=False)
    training_pairs.to_parquet(f"{output_dir}/train_pairs.parquet", index=False)
    
    # Save encoders
    with open(f"{output_dir}/product_encoder.pkl", 'wb') as f:
        pickle.dump(product_encoder, f)
    
    with open(f"{output_dir}/brand_encoder.pkl", 'wb') as f:
        pickle.dump(brand_encoder, f)
    
    # with open(f"{output_dir}/user_encoder.pkl", 'wb') as f:
    #     pickle.dump(user_encoder, f)
    
    logger.info("Data saved successfully!")

def main():
    """Main processing pipeline"""
    # File paths
    review_file = "dataset/AMAZON_FASHION.json.gz"
    meta_file = "dataset/meta_AMAZON_FASHION.json.gz"
    output_dir = "processed_data"
    
    # Load data
    review_df = load_review_data(review_file)
    product_df = load_product_metadata(meta_file)
    
    # Process features
    item_features = process_item_features(product_df)
    user_features = process_user_features(review_df, min_purchases=3)
    
    # Create encoders
    product_encoder, brand_encoder = create_label_encoders(item_features, user_features)
    
    # Encode features
    encoded_item_features = encode_item_features(item_features, product_encoder, brand_encoder)
    encoded_user_features = encode_user_features(user_features, product_encoder)
    
    # Create training pairs
    training_pairs = create_training_pairs(encoded_user_features, product_encoder)
    
    # Save all data
    save_data(encoded_item_features, encoded_user_features, training_pairs,
              product_encoder, brand_encoder, output_dir)
    
    # Print summary
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Items processed: {len(encoded_item_features)}")
    print(f"Users processed: {len(encoded_user_features)}")
    print(f"Training pairs created: {len(training_pairs)}")
    print(f"Unique products: {len(product_encoder.classes_)}")
    print(f"Unique brands: {len(brand_encoder.classes_)}")
    # print(f"Unique users: {len(user_encoder.classes_)}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()