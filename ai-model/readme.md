# Recommendation System

This project contains two recommendation approaches:

1. **Two-Tower Architecture** - Deep learning model with User Tower and Product Tower
2. **Content-Based Filtering** - Direct content similarity using text embeddings in Milvus

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
├── data/                              # Data processing scripts
│   ├── database_config.py             # PostgreSQL connection configuration
│   ├── encoder.py                     # SentenceTransformer text encoding
│   ├── process_data.py                # Raw data → processed parquet files
│   ├── stream_meta_to_postgres.py     # Import metadata into PostgreSQL
│   ├── compute_product_vectors.py     # Generate product embeddings (Two-Tower)
│   ├── actual_milvus_integration.py   # Milvus collection management
│   ├── generate_content_test_set.py   # Generate test set for content-based (NEW)
│   └── dataset/
│       └── products_clean.csv         # Cleaned product data
├── main/                              # Core model code
│   ├── product_tower.py               # Product Tower model
│   ├── user_tower.py                  # User Tower model
│   ├── train.py                       # Training script (Two-Tower)
│   ├── inference_pipeline.py          # Recommendation pipeline
│   ├── evaluate.py                    # Evaluation (Two-Tower)
│   ├── evaluate_content_based.py      # Evaluation (Content-Based) (NEW)
│   └── util/
│       └── util.py                    # Utility functions
├── docker-compose.yml                 # Milvus services
├── requirements.txt                   # Python dependencies
├── run_content_based.sh               # Content-based pipeline (Linux/Mac)
├── run_content_based.bat              # Content-based pipeline (Windows)
└── readme.md                          # This file
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

---

## Content-Based Filtering (No User Tower Required)

This approach uses **content similarity** directly via text embeddings stored in Milvus.
It does NOT require training a user tower model — recommendations are based purely on
product text content similarity.

### How It Works

```
Product Title/Description ──► SentenceTransformer ──► Embedding (384-d)
                                                            │
                                                   Insert into Milvus
                                                            │
Query Text ──► SentenceTransformer ──► Embedding ──► ANN Search ──► Similar Products
```

### Quick Start - Content-Based Filtering

#### Step 1: Generate Test Set

```bash
cd ai-model
python data/generate_content_test_set.py \
    --csv-path data/dataset/products_clean.csv \
    --output-dir data/processed_data \
    --n-queries 1000
```

This script:
- Loads products from `products_clean.csv`
- Computes text embeddings using SentenceTransformer
- Creates query/ground-truth pairs for evaluation
- Saves test set to `data/processed_data/`

Output files:
- `content_test_queries.parquet` - Test queries with ground truth
- `content_products.parquet` - Product catalog
- `content_embeddings.npy` - Pre-computed embeddings
- `content_test_metadata.pkl` - Test set statistics

#### Step 2: Evaluate Content-Based Filtering

```bash
# Brute-force evaluation (recommended for accuracy)
python main/evaluate_content_based.py --data-dir data/processed_data

# Using Milvus ANN search (faster for large catalogs)
python main/evaluate_content_based.py --data-dir data/processed_data --use-milvus

# Sample mode for quick testing
python main/evaluate_content_based.py --data-dir data/processed_data --sample-queries 100
```

#### Step 3: Windows One-Click Run

Simply double-click `run_content_based.bat` in the `ai-model` folder.

#### Linux/Mac One-Command Run

```bash
cd ai-model
chmod +x run_content_based.sh
./run_content_based.sh
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of relevant (same-brand similar) items found in top-K |
| **NDCG@K** | Ranking quality — higher if relevant items appear earlier |
| **Hit Rate@K** | % of queries with at least one relevant item in top-K |
| **MRR@K** | Mean Reciprocal Rank — how early the first relevant item appears |
| **Item Coverage** | % of catalog items that get recommended |

### Configuration

The test set generation can be configured:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv-path` | `data/dataset/products_clean.csv` | Path to cleaned products |
| `--output-dir` | `data/processed_data` | Where to save test set |
| `--n-queries` | `1000` | Number of test queries |
| `--seed` | `42` | Random seed for reproducibility |

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