# Two-Tower Recommendation System

A deep learning-based recommendation system using the **Two-Tower architecture** with contrastive learning, implemented in PyTorch. Product embeddings are stored in **Milvus** for fast similarity search at inference time.

## Architecture

```
┌──────────────────────┐     ┌──────────────────────┐
│     User Tower       │     │    Product Tower     │
│                      │     │                      │
│  Product History ──► │     │  Text Embedding ──►  │
│  (GRU Encoder)       │     │  Brand Embedding     │
│                      │     │  Price Embedding     │
│     User Vector ◄─── │     │  Combined MLP ──►    │
│        (64-d)        │     │   Product Vector     │
└──────────┬───────────┘     │        (64-d)        │
           │                 └──────────┬───────────┘
           │                            │
           │      Inner Product         │
           │     (Similarity)           │
           └────────────┬───────────────┘
                        ▼
              Recommendation Score
```

### Components

| Component | Description |
|-----------|------------|
| **User Tower** | GRU-based encoder that processes a user's product interaction history |
| **Product Tower** | Multi-modal encoder combining text (SentenceTransformer), brand embeddings, and price features |
| **Contrastive Loss** | In-batch negative sampling with temperature-scaled cross-entropy |
| **Milvus** | Vector database for storing and searching product embeddings (IVF_FLAT index) |

## Project Structure

```
ai-model/
├── data/                          # Data processing scripts
│   ├── database_config.py         # PostgreSQL connection configuration
│   ├── encoder.py                 # SentenceTransformer text encoding
│   ├── process_data.py            # Raw data → processed parquet files
│   ├── stream_meta_to_postgres.py # Import metadata into PostgreSQL
│   ├── compute_product_vectors.py # Generate product embeddings
│   └── actual_milvus_integration.py # Milvus collection management
├── main/                          # Core model code
│   ├── product_tower.py           # Product Tower model
│   ├── user_tower.py              # User Tower model
│   ├── train.py                   # Training script
│   ├── inference_pipeline.py      # Recommendation pipeline
│   └── util/
│       └── util.py                # Utility functions
├── docker-compose.yml             # Milvus services
├── requirements.txt               # Python dependencies
└── readme.md                      # This file
```

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Start Milvus

```bash
docker compose up -d
```

### 3. Data Pipeline

**Step 3a: Import product metadata into PostgreSQL**
```bash
cd data
python stream_meta_to_postgres.py
```

**Step 3b: Process raw Amazon Fashion data**
```bash
cd data
python process_data.py
```
This generates:
- `processed_data/item_features.parquet`
- `processed_data/user_features.parquet`
- `processed_data/train_pairs.parquet`
- `processed_data/brand_encoder.pkl`
- `processed_data/product_encoder.pkl`

### 4. Train the Model

```bash
cd main
python train.py
```
Checkpoints are saved to `model/user_tower_checkpoint.pth` and `model/product_tower_checkpoint.pth`.

### 5. Compute Product Vectors

```bash
cd data
python compute_product_vectors.py   # Standalone vector computation
# or
python actual_milvus_integration.py # Compute + load into Milvus
```

### 6. Evaluate the Model

```bash
cd main
python evaluate.py
```

**Options:**

```bash
# Evaluate on a sample of users for faster results
python evaluate.py --sample-users 1000

# Custom K values
python evaluate.py --k-values 5 10 20 50

# Specify device
python evaluate.py --device cpu
```

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of user's test items found in top-K |
| **NDCG@K** | Ranking quality — higher if relevant items appear earlier |
| **Hit Rate@K** | % of users with at least one relevant item in top-K |
| **MRR@K** | Mean Reciprocal Rank — how early the first relevant item appears |
| **Item Coverage** | % of catalog items that get recommended |

### 7. Serve Recommendations

```bash
cd main
python inference_pipeline.py
```

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `recommendation_db` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | `password123` | Database password |
| `MILVUS_HOST` | `localhost` | Milvus server host |
| `MILVUS_PORT` | `19530` | Milvus server port |
| `MILVUS_USER` | `root` | Milvus username |
| `MILVUS_PASSWORD` | | Milvus password |

### Training Hyperparameters

Edit `TrainingConfig` in `main/train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | `256` | Training batch size |
| `NUM_EPOCHS` | `20` | Number of training epochs |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `OUTPUT_DIM` | `64` | Embedding dimension |
| `TEMPERATURE` | `0.07` | Contrastive loss temperature |
| `MAX_GRAD_NORM` | `1.0` | Gradient clipping threshold |

## Dataset

This project uses the **Amazon Fashion** dataset:
- **Reviews**: `AMAZON_FASHION.json.gz` — user interactions with timestamps
- **Metadata**: `meta_AMAZON_FASHION.json.gz` — product details (title, brand, description, features)

Place both files in `ai-model/data/dataset/`.

## Reference

> Jianmo Ni, Jiacheng Li, Julian McAuley. *Justifying recommendations using distantly-labeled reviews and fined-grained aspects.* EMNLP, 2019.