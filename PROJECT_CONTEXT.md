# Project Context — Recommendation System

## Overview
A recommendation system with two components:
- **ai-model/** — Python ML service using SentenceTransformer (all-MiniLM-L6-v2) for content-based filtering. Stores 384-dim embeddings in Milvus (IVF_FLAT, IP metric on L2-normalised vectors).
- **flink/** — Java Flink streaming job that processes real-time user product view events from Kafka, aggregates them per user in rolling windows, and outputs user interest profiles to a downstream Kafka topic.

## Data Flow (End-to-End)
```
Java Backend ──► Kafka: user-view-events ──► Flink (aggregate) ──► Kafka: user-interest-profiles ──► ML Service
```

## Product Identity (Critical)
- `products_clean.csv` uses **ASIN** strings (e.g. `"0764443682"`, `"630456984X"`) as product identifiers.
- Milvus collection `content_based_products` stores:
  - **`product_id`** (VARCHAR) — the original ASIN. **This is what the backend and Flink must use.**
  - **`product_idx`** (INT64) — row position in the training split. Internal-only, do not expose externally.
- Flink `UserViewEvent.productId` must be the **ASIN**, not the `product_idx`.

## Key Decisions / Gotchas

**Flink Kafka connector — internal Docker networking:**
Inside Flink containers, Kafka bootstrap servers must be `kafka:29092` (Docker DNS), not `localhost:9092`. The `localhost:9092` listener is only for the host machine.

**Flink Docker image uses Java 17:**
The `flink:1.18-java17` image only supports Java 17 class files (version 61.0). Even though the dev machine has Java 24, compile with:
```xml
<maven.compiler.release>17</maven.compiler.release>
```

**Lombok incompatible with Java 24:**
Removed Lombok. Model classes are plain POJOs.

**Window:** Currently set to **1 minute** (processing time) for testing. Production should use `60` minutes.

## Flink JAR Pipeline
1. `mvn clean package` in `flink/`
2. Upload via: `curl -X POST -H "Expect:" -F "jarfile=@target/flink-job-1.0-SNAPSHOT.jar" http://localhost:8081/jars/upload`
3. Run via: `curl -X POST http://localhost:8081/jars/{jar_id}_flink-job-1.0-SNAPSHOT.jar/run`
4. Cancel via: `wsl docker exec flink-jobmanager flink cancel {job_id}`

## Docker Compose Services (flink/docker-compose.yml)
| Service | Port | Image |
|---------|------|-------|
| zookeeper | 2181 | confluentinc/cp-zookeeper:7.5.0 |
| kafka | 9092 (host), 29092 (internal) | confluentinc/cp-kafka:7.5.0 |
| jobmanager | 8081 | flink:1.18-java17 |
| taskmanager | — | flink:1.18-java17 |

## AI Model Stack (ai-model/)
- **Milvus**: `localhost:19530`, auth: `root` / `Milvus`
- **Embedding**: all-MiniLM-L6-v2, 384-dim, L2-normalised
- **Collection**: `content_based_products`
- **Milvus docker**: managed by `ai-model/docker-compose.yml` (separate from Flink compose)

## Kafka Topics
| Topic | Direction | Schema |
|-------|-----------|--------|
| `user-view-events` | In (from Java backend) | `{"userId": "...", "productId": "ASIN", "timestamp": long}` |
| `user-interest-profiles` | Out (from Flink) | `{"userId": "...", "viewedProducts": {"ASIN": count}, "windowStart": long, "windowEnd": long}` |

## Testing
```bash
# Send test events (uses real ASINs from products_clean.csv)
cd flink/
python test_produce_events.py

# Read output (after window fires)
docker exec flink-kafka bash -c 'kafka-console-consumer --bootstrap-server localhost:9092 --topic user-interest-profiles --from-beginning --max-messages 10 --timeout-ms 10000'
```

## Backend Integration Notes
- Java backend should publish `UserViewEvent` JSON to `kafka:29092` (from within Docker) or `localhost:9092` (from host).
- Same format as `test_produce_events.py`: userId, productId (ASIN), timestamp (epoch millis).
- Backend `Product` model: `asin` (String, PK), `title`, `description`, `brand`, `imageURL`.
