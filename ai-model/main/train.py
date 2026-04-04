"""Training script for the Two-Tower recommendation model.

Implements contrastive learning with in-batch negative sampling
to jointly train UserTower and ProductTower networks.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Allow imports from current directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.product_tower import ProductTower
from main.user_tower import UserTower

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────


class TrainingConfig:
    """Production hyperparameters for training.

    Tuned for 186K product catalog with 35K training pairs.
    Key insight: With large catalog and sparse data, we need:
    - Smaller model to avoid overfitting
    - Higher temperature for smoother gradients
    - Lower learning rate for stable training
    - More epochs for convergence
    """

    # Data paths (relative to project root ai-model/)
    DATA_DIR: str = "data/processed_data"
    MODEL_DIR: str = "model"

    # Model architecture - smaller to match data size
    USER_PRODUCT_EMBED_DIM: int = 32
    USER_HIDDEN_DIM: int = 128
    USER_NUM_LAYERS: int = 2
    PRODUCT_TEXT_DIM: int = 384
    PRODUCT_BRAND_EMBED_DIM: int = 16
    PRODUCT_PRICE_EMBED_DIM: int = 16
    PRODUCT_HIDDEN_DIM: int = 128
    OUTPUT_DIM: int = 64

    # Training hyperparameters
    BATCH_SIZE: int = 256
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    MAX_GRAD_NORM: float = 1.0
    TEMPERATURE: float = 0.1  # Higher temperature for smoother gradients

    # Scheduler
    SCHEDULER_STEP_SIZE: int = 15
    SCHEDULER_GAMMA: float = 0.5

    # Data
    MAX_HISTORY_LENGTH: int = 50
    DATALOADER_NUM_WORKERS: int = 0  # Set to >0 on Linux for faster loading


# ── Dataset ──────────────────────────────────────────────


class TwoTowerDataset(Dataset):
    """Dataset that yields (user_history, target_item) positive pairs.

    Each sample contains a user's interaction history and the target item
    they interacted with, used for contrastive learning.
    """

    def __init__(
        self,
        train_pairs_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        text_embeddings: np.ndarray,
    ) -> None:
        self.pairs = train_pairs_df.reset_index(drop=True)
        self.item_df = item_features_df.reset_index(drop=True)
        self.text_embs = text_embeddings  # np.array of shape (num_items, 384)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.pairs.iloc[idx]

        # User history
        history = torch.tensor(row["context_history"], dtype=torch.long)

        # Target item features
        item_idx = int(row["target_item_id"])
        item_row = self.item_df.iloc[item_idx]

        text_emb = torch.tensor(self.text_embs[item_idx], dtype=torch.float32)
        brand_id = torch.tensor(item_row["brand_id"], dtype=torch.long)
        price = torch.tensor(0.0, dtype=torch.float32)  # Replace with actual price if available

        return {
            "history": history,
            "text_emb": text_emb,
            "brand_id": brand_id,
            "price": price,
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad histories to the same length within a batch."""
    histories = [item["history"] for item in batch]
    histories_padded = torch.nn.utils.rnn.pad_sequence(
        histories, batch_first=True, padding_value=0
    )

    return {
        "history": histories_padded,
        "text_emb": torch.stack([item["text_emb"] for item in batch]),
        "brand_id": torch.stack([item["brand_id"] for item in batch]),
        "price": torch.stack([item["price"] for item in batch]),
    }


# ── Loss: In-batch Negative Sampling ────────────────────


def contrastive_loss(
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute contrastive loss using other items in the batch as negatives.

    Args:
        user_vectors: User embeddings of shape (batch_size, output_dim).
        item_vectors: Item embeddings of shape (batch_size, output_dim).
        temperature: Temperature scaling for similarity computation.

    Returns:
        Cross-entropy loss over the similarity matrix.
    """
    # Similarity matrix (batch_size, batch_size)
    similarity = torch.matmul(user_vectors, item_vectors.T) / temperature

    # Labels: diagonal entries are positive pairs
    labels = torch.arange(similarity.size(0), device=similarity.device)

    return torch.nn.functional.cross_entropy(similarity, labels)


# ── Training Loop ────────────────────────────────────────


def load_or_compute_text_embeddings(
    item_features_df: pd.DataFrame, output_path: Path
) -> np.ndarray:
    """Load cached text embeddings or compute them using SentenceTransformer."""
    if output_path.exists():
        logger.info("Loading cached text embeddings...")
        return np.load(output_path)

    logger.info("Computing text embeddings (first time)...")
    from sentence_transformers import SentenceTransformer

    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [
        f"{row['title']} {row['description']}"
        for _, row in item_features_df.iterrows()
    ]
    embeddings = text_model.encode(texts, batch_size=128, show_progress_bar=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    return embeddings


def train(data_dir: str | None = None, model_dir: str | None = None) -> None:
    """Run the full training pipeline.

    Args:
        data_dir: Path to the processed data directory.
        model_dir: Path to save model checkpoints.
    """
    config = TrainingConfig()

    # Override paths if provided
    if data_dir:
        config.DATA_DIR = data_dir
    if model_dir:
        config.MODEL_DIR = model_dir

    data_path = Path(config.DATA_DIR)
    model_path = Path(config.MODEL_DIR)
    model_path.mkdir(parents=True, exist_ok=True)

    logger.info("=== STARTING TRAINING ===")

    # ── Load data ────────────────────────────────────────
    train_pairs = pd.read_parquet(data_path / "train_pairs.parquet")
    user_features = pd.read_parquet(data_path / "user_features.parquet")
    item_features = pd.read_parquet(data_path / "item_features.parquet")

    with open(data_path / "product_encoder.pkl", "rb") as f:
        product_encoder = pickle.load(f)
    with open(data_path / "brand_encoder.pkl", "rb") as f:
        brand_encoder = pickle.load(f)

    # ── Text embeddings ──────────────────────────────────
    text_emb_path = data_path / "text_embeddings.npy"
    text_embeddings = load_or_compute_text_embeddings(item_features, text_emb_path)

    # ── Dataset & DataLoader ─────────────────────────────
    dataset = TwoTowerDataset(train_pairs, item_features, text_embeddings)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.DATALOADER_NUM_WORKERS,
    )

    logger.info("Dataset size: %d pairs", len(dataset))
    logger.info("Unique products: %d", len(product_encoder.classes_))
    logger.info("Unique brands: %d", len(brand_encoder.classes_))

    # ── Models ───────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

    user_tower = UserTower(
        product_vocab_size=len(product_encoder.classes_),
        product_embed_dim=config.USER_PRODUCT_EMBED_DIM,
        hidden_dim=config.USER_HIDDEN_DIM,
        num_layers=config.USER_NUM_LAYERS,
        output_dim=config.OUTPUT_DIM,
    ).to(device)

    product_tower = ProductTower(
        brand_vocab_size=len(brand_encoder.classes_),
        text_dim=config.PRODUCT_TEXT_DIM,
        brand_embed_dim=config.PRODUCT_BRAND_EMBED_DIM,
        price_embed_dim=config.PRODUCT_PRICE_EMBED_DIM,
        hidden_dim=config.PRODUCT_HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
    ).to(device)

    # Count parameters
    user_params = sum(p.numel() for p in user_tower.parameters())
    product_params = sum(p.numel() for p in product_tower.parameters())
    logger.info("UserTower parameters: %s", f"{user_params:,}")
    logger.info("ProductTower parameters: %s", f"{product_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────
    optimizer = torch.optim.Adam(
        list(user_tower.parameters()) + list(product_tower.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.SCHEDULER_STEP_SIZE,
        gamma=config.SCHEDULER_GAMMA,
    )

    # ── Training loop ────────────────────────────────────
    best_loss = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        user_tower.train()
        product_tower.train()

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            history = batch["history"].to(device)
            text_emb = batch["text_emb"].to(device)
            brand_id = batch["brand_id"].to(device)
            price = batch["price"].to(device)

            # Forward pass
            user_vec = user_tower(history)
            product_vec = product_tower(text_emb, brand_id, price)

            # Compute loss
            loss = contrastive_loss(user_vec, product_vec, temperature=config.TEMPERATURE)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(user_tower.parameters()) + list(product_tower.parameters()),
                max_norm=config.MAX_GRAD_NORM,
            )
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        scheduler.step()

        logger.info(
            "Epoch %d/%d | Loss: %.4f | LR: %.6f",
            epoch + 1,
            config.NUM_EPOCHS,
            avg_loss,
            scheduler.get_last_lr()[0],
        )

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": user_tower.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                model_path / "user_tower_checkpoint.pth",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": product_tower.state_dict(),
                    "loss": best_loss,
                },
                model_path / "product_tower_checkpoint.pth",
            )
            logger.info("  Saved best checkpoint (loss: %.4f)", best_loss)

    logger.info("=== TRAINING COMPLETE | Best loss: %.4f ===", best_loss)


if __name__ == "__main__":
    train()
