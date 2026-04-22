# 🎯 Recommendation System

A modern, microservices-based product recommendation engine built with **Spring Boot**, **Apache Flink**, **React**, and **Machine Learning**. The system provides personalized product recommendations using content-based filtering, real-time user behavior tracking with Kafka, and trending product analysis via Flink stream processing.

---

## 🏗️ Architecture Overview

![System Architecture](./images/Screenshot%20from%202026-04-22%2017-42-01.png)

The system consists of four main components:

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Backend** | Spring Boot 3 + PostgreSQL + Redis + Kafka | REST API, authentication, data management |
| **Frontend** | React + TypeScript + Vite | User interface for browsing and recommendations |
| **AI Model** | Python + Sentence Transformers | Content-based filtering & recommendation evaluation |
| **Stream Processor** | Apache Flink | Real-time user view event aggregation & trending |

![Recommendation Flow](./images/Screenshot%20from%202026-04-22%2017-43-21.png)

---

## 🚀 Tech Stack

### Backend
- **Java 21** + **Spring Boot 3**
- **PostgreSQL** — Primary database
- **Redis** — Caching & recommendation storage
- **Apache Kafka** — Event streaming
- **JWT** — Authentication
- **Swagger/OpenAPI** — API documentation

### Frontend
- **React 18** + **TypeScript**
- **Vite** — Build tool
- **React Router** — Client-side routing
- **Axios** — HTTP client

### Stream Processing
- **Apache Flink** — Real-time event aggregation
- **Kafka** — Event source & sink
- **Redis** — Recommendation result store

### AI / Machine Learning
- **Python** — Model training & evaluation
- **Sentence Transformers** — Text embeddings
- **Content-Based Filtering** — Recommendation algorithm

---

## 📦 Prerequisites

- **Java 21**
- **Maven 3.6+**
- **Node.js 18+** & **npm**
- **Python 3.10+**
- **Docker & Docker Compose** (for containerized deployment)

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/VuDance/recommendation-system.git
cd recommendation-system
```

### 2. Start Infrastructure Services

```bash
docker-compose -f back-end/compose.yaml up -d
```

This starts PostgreSQL, Redis, and Kafka.

### 3. Start the Backend

```bash
cd back-end
./mvnw spring-boot:run
```

The API will be available at **http://localhost:8080**.

### 4. Start the Frontend

```bash
cd front-end/vite-project
npm install
npm run dev
```

The UI will be available at **http://localhost:5173**.

### 5. AI Model — Generate Test Set & Evaluate

```bash
cd ai-model

# Generate test set
python data/generate_content_test_set.py \
    --csv-path data/dataset/products_clean.csv \
    --output-dir data/processed_data \
    --n-queries 1000

# Evaluate
# Evaluate on a sample of users for faster results
python evaluate.py --sample-users 1000

# Custom K values
python evaluate.py --k-values 5 10 20 50
```

### 6. Stream Processing (Flink)

```bash
cd flink
# Build & submit the Flink job
./scripts/submit-job.sh
```

---

## 🔌 API Endpoints

### Authentication (`/api/auth`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/login` | Login with email & password, returns JWT token |
| `POST` | `/api/auth/register` | Register a new user account |

### Products (`/api/products`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/products?page=0&size=10` | Get all products with pagination |
| `GET` | `/api/products/{id}` | Get product by ID (also logs view event) |
| `GET` | `/api/products/category/{category}` | Get products by category |
| `GET` | `/api/products/search?keyword=` | Search products by keyword |

### Recommendations (`/api/recommendations`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/recommendations/{userId}` | Get personalized recommendations for a user |

### Swagger Documentation

- **Swagger UI**: http://localhost:8080/swagger-ui.html
- **OpenAPI JSON**: http://localhost:8080/v3/api-docs

---

---

## 🧪 AI Model Details

The recommendation engine uses **Content-Based Filtering** with the following metrics:

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of user's test items found in top-K |
| **NDCG@K** | Ranking quality — higher if relevant items appear earlier |
| **Hit Rate@K** | % of users with at least one relevant item in top-K |
| **MRR@K** | Mean Reciprocal Rank — how early the first relevant item appears |
| **Item Coverage** | % of catalog items that get recommended |

### Dataset

Uses the **Amazon Fashion** dataset (metadata: `meta_AMAZON_FASHION.json.gz`).

> Jianmo Ni, Jiacheng Li, Julian McAuley. *Justifying recommendations using distantly-labeled reviews and fined-grained aspects.* EMNLP, 2019.

---

## 📄 License

This project is licensed under the **MIT License**.
