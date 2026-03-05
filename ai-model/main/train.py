#!/usr/bin/env python3
"""
Training script for Two-Tower model
"""
import pandas as pd
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from product_tower import ProductTower
from user_tower import UserTower

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Dataset ─────────────────────────────────────────────

class TwoTowerDataset(Dataset):
    """
    Mỗi sample là 1 cặp (user_history, target_item) — positive pair
    """
    def __init__(self, train_pairs_df, user_features_df, item_features_df, text_embeddings):
        self.pairs        = train_pairs_df.reset_index(drop=True)
        self.user_df      = user_features_df.set_index('reviewerID')
        self.item_df      = item_features_df.reset_index(drop=True)
        self.text_embs    = text_embeddings  # np.array (num_items, 384)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        # User history
        history = torch.tensor(row['context_history'], dtype=torch.long)

        # Target item features
        item_idx  = int(row['target_item_id'])
        item_row  = self.item_df.iloc[item_idx]

        text_emb  = torch.tensor(self.text_embs[item_idx], dtype=torch.float32)
        brand_id  = torch.tensor(item_row['brand_id'], dtype=torch.long)
        price     = torch.tensor(0.0, dtype=torch.float32)  # thay bằng giá thực nếu có

        return {
            'history'  : history,
            'text_emb' : text_emb,
            'brand_id' : brand_id,
            'price'    : price,
        }


def collate_fn(batch):
    """Padding history về cùng độ dài trong batch"""
    histories = [b['history'] for b in batch]
    histories_padded = torch.nn.utils.rnn.pad_sequence(
        histories, batch_first=True, padding_value=0
    )
    return {
        'history'  : histories_padded,
        'text_emb' : torch.stack([b['text_emb'] for b in batch]),
        'brand_id' : torch.stack([b['brand_id'] for b in batch]),
        'price'    : torch.stack([b['price']    for b in batch]),
    }


# ── Loss: In-batch Negative Sampling ────────────────────

def contrastive_loss(user_vectors, item_vectors, temperature=0.07):
    """
    Dùng các item khác trong batch làm negative samples
    user_vectors : (B, 64)
    item_vectors : (B, 64)
    """
    # Similarity matrix (B, B)
    similarity = torch.matmul(user_vectors, item_vectors.T) / temperature

    # Label: diagonal là positive pair
    labels = torch.arange(similarity.size(0), device=similarity.device)

    loss = torch.nn.functional.cross_entropy(similarity, labels)
    return loss


# ── Training ─────────────────────────────────────────────

def train():
    logger.info("=== STARTING TRAINING ===")

    # Load data
    train_pairs   = pd.read_parquet('../data/processed_data/train_pairs.parquet')
    user_features = pd.read_parquet('../data/processed_data/user_features.parquet')
    item_features = pd.read_parquet('../data/processed_data/item_features.parquet')

    with open('../data/processed_data/product_encoder.pkl', 'rb') as f:
        product_encoder = pickle.load(f)
    with open('../data/processed_data/brand_encoder.pkl', 'rb') as f:
        brand_encoder = pickle.load(f)

    # Load hoặc compute text embeddings
    text_emb_path = Path('../data/processed_data/text_embeddings.npy')
    if text_emb_path.exists():
        logger.info("Loading cached text embeddings...")
        text_embeddings = np.load(text_emb_path)
    else:
        logger.info("Computing text embeddings (first time)...")
        from sentence_transformers import SentenceTransformer
        text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        texts = [
            f"{r['title']} {r['description']}"
            for _, r in item_features.iterrows()
        ]
        text_embeddings = text_model.encode(texts, batch_size=128, show_progress_bar=True)
        np.save(text_emb_path, text_embeddings)

    # Dataset & DataLoader
    dataset    = TwoTowerDataset(train_pairs, user_features, item_features, text_embeddings)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True,
                            collate_fn=collate_fn, num_workers=4)

    logger.info(f"Dataset size: {len(dataset)} pairs")

    # Models
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user_tower   = UserTower(product_vocab_size=len(product_encoder.classes_)).to(device)
    product_tower = ProductTower(brand_vocab_size=len(brand_encoder.classes_)).to(device)

    logger.info(f"Training on: {device}")

    # Optimizer
    optimizer = torch.optim.Adam(
        list(user_tower.parameters()) + list(product_tower.parameters()),
        lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    num_epochs = 20
    best_loss  = float('inf')

    Path('model').mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        user_tower.train()
        product_tower.train()

        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            history   = batch['history'].to(device)
            text_emb  = batch['text_emb'].to(device)
            brand_id  = batch['brand_id'].to(device)
            price     = batch['price'].to(device)

            # Forward
            user_vec    = user_tower(history)
            product_vec = product_tower(text_emb, brand_id, price)

            # Loss
            loss = contrastive_loss(user_vec, product_vec)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(user_tower.parameters()) + list(product_tower.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch'              : epoch,
                'model_state_dict'   : user_tower.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss'               : best_loss,
            }, 'model/user_tower_checkpoint.pth')

            torch.save({
                'epoch'              : epoch,
                'model_state_dict'   : product_tower.state_dict(),
                'loss'               : best_loss,
            }, 'model/product_tower_checkpoint.pth')

            logger.info(f"  ✅ Saved best checkpoint (loss: {best_loss:.4f})")

    logger.info(f"=== TRAINING COMPLETE | Best loss: {best_loss:.4f} ===")


if __name__ == "__main__":
    train()