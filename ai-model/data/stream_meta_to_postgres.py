"""Stream processor for Amazon product metadata into PostgreSQL.

Reads gzipped JSONL files (e.g., meta_AMAZON_FASHION.json.gz),
extracts product fields, and bulk-inserts them into a PostgreSQL
table with progress tracking and error handling.
"""

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import psycopg2
from psycopg2.extras import execute_values

from data.database_config import DatabaseConfig

# ── Logging Configuration ─────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("stream_meta_to_postgres.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Product Data Extraction ──────────────────────────────

DESCRIPTION_FIELDS = (
    "description",
    "feature",
    "details",
    "product_description",
)


def extract_product_data(json_line: str) -> Optional[dict[str, Any]]:
    """Extract product metadata from a single JSON line.

    Args:
        json_line: A single JSONL line from the meta file.

    Returns:
        Dict with ASIN, title, brand, image_url, description;
        or None if parsing fails or ASIN is missing.
    """
    try:
        data = json.loads(json_line.strip())
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON line: %s", exc)
        return None

    asin = data.get("asin")
    if not asin:
        return None

    return {
        "asin": asin,
        "title": data.get("title", ""),
        "brand": data.get("brand", ""),
        "image_url": data.get("imageURL", ""),
        "description": _extract_description(data),
    }


def _extract_description(data: dict[str, Any]) -> str:
    """Extract a description string from various possible fields."""
    for field in DESCRIPTION_FIELDS:
        if field in data:
            value = data[field]
            if isinstance(value, list):
                return " ".join(str(item) for item in value if item)
            if isinstance(value, str):
                return value
            if value:
                return str(value)

    # Fall back to title if it's unusually long (may contain description)
    title = data.get("title", "")
    if len(title) > 200:
        return title

    return ""


# ── Streamer Class ───────────────────────────────────────


class MetaDataStreamer:
    """Streams product metadata from a gzipped JSONL file into PostgreSQL.

    Args:
        db_config: DatabaseConfig instance with connection settings.
    """

    def __init__(self, db_config: DatabaseConfig) -> None:
        self.db_config = db_config
        self.connection: psycopg2.extensions.connection | None = None
        self.cursor: psycopg2.extensions.cursor | None = None
        self.total_processed = 0
        self.total_inserted = 0
        self.total_skipped = 0
        self.batch_size = db_config.BATCH_SIZE

    # ── Database Management ──────────────────────────────

    def connect_to_database(self) -> bool:
        """Establish a PostgreSQL connection."""
        try:
            self.connection = psycopg2.connect(**self.db_config.get_connection_params())
            self.cursor = self.connection.cursor()
            logger.info("Connected to PostgreSQL database.")
            return True
        except Exception as exc:
            logger.error("Database connection failed: %s", exc)
            return False

    def create_table(self) -> None:
        """Create the products table if it does not exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS products (
            asin VARCHAR(20) PRIMARY KEY,
            title TEXT NOT NULL,
            brand VARCHAR(255),
            image_url TEXT,
            description TEXT
        );
        """
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("Products table created successfully.")
        except Exception as exc:
            logger.error("Failed to create table: %s", exc)
            self.connection.rollback()
            raise

    def close_connection(self) -> None:
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed.")

    # ── Batch Insertion ─────────────────────────────────

    def insert_batch(self, batch_data: list[tuple]) -> bool:
        """Insert a batch of product records.

        Args:
            batch_data: List of (asin, title, brand, image_url, description) tuples.

        Returns:
            True on success, False on failure.
        """
        if not batch_data:
            return True

        insert_query = """
        INSERT INTO products (asin, title, brand, image_url, description)
        VALUES %%s
        ON CONFLICT (asin) DO NOTHING
        """

        try:
            execute_values(
                self.cursor,
                insert_query,
                batch_data,
                template=None,
                page_size=len(batch_data),
            )
            self.connection.commit()
            self.total_inserted += len(batch_data)
            return True
        except Exception as exc:
            logger.error("Batch insert failed: %s", exc)
            self.connection.rollback()
            return False

    # ── Streaming Pipeline ──────────────────────────────

    def stream_and_insert(
        self,
        file_path: str | Path,
        max_rows: int | None = None,
    ) -> bool:
        """Stream from a gzipped JSONL file and bulk-insert into PostgreSQL.

        Args:
            file_path: Path to the gzipped JSONL file.
            max_rows: Maximum number of rows to process (None for unlimited).

        Returns:
            True if processing completed successfully.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            return False

        logger.info("Processing file: %s", file_path.name)
        logger.info("Batch size: %d", self.batch_size)
        logger.info("Max rows: %s", max_rows or "All")

        start_time = time.time()
        batch_data: list[tuple] = []

        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if max_rows and self.total_processed >= max_rows:
                        break

                    product_data = extract_product_data(line)
                    if product_data:
                        batch_data.append((
                            product_data["asin"],
                            product_data["title"],
                            product_data["brand"],
                            product_data["image_url"],
                            product_data["description"],
                        ))

                        if len(batch_data) >= self.batch_size:
                            if self.insert_batch(batch_data):
                                self.total_processed += len(batch_data)
                                logger.info(
                                    "Processed: %s, Inserted: %s, Skipped: %s",
                                    self.total_processed,
                                    self.total_inserted,
                                    self.total_skipped,
                                )
                            else:
                                self.total_skipped += len(batch_data)
                            batch_data = []
                    else:
                        self.total_skipped += 1

                    if line_num % 10000 == 0:
                        elapsed = time.time() - start_time
                        logger.info(
                            "Progress: %s lines processed (%.2fs elapsed)",
                            f"{line_num:,}",
                            elapsed,
                        )

        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user.")
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            return False

        # Insert final batch
        if batch_data:
            if self.insert_batch(batch_data):
                self.total_processed += len(batch_data)
                logger.info(
                    "Final batch — Processed: %s, Inserted: %s",
                    self.total_processed,
                    self.total_inserted,
                )
            else:
                self.total_skipped += len(batch_data)

        elapsed = time.time() - start_time
        logger.info("Processing completed!")
        logger.info("Total lines processed: %s", f"{self.total_processed:,}")
        logger.info("Total records inserted: %s", f"{self.total_inserted:,}")
        logger.info("Total records skipped: %s", f"{self.total_skipped:,}")
        logger.info("Total time: %.2f seconds", elapsed)
        if elapsed > 0:
            logger.info(
                "Average speed: %.2f lines/second",
                self.total_processed / elapsed,
            )

        return True

    # ── Statistics ──────────────────────────────────────

    def get_table_stats(self) -> None:
        """Log statistics about the products table."""
        if not self.cursor:
            logger.warning("No database connection for stats query.")
            return

        try:
            queries = {
                "Total records": "SELECT COUNT(*) FROM products",
                "Records with title": "SELECT COUNT(*) FROM products WHERE title IS NOT NULL AND title != ''",
                "Records with brand": "SELECT COUNT(*) FROM products WHERE brand IS NOT NULL AND brand != ''",
                "Records with image": "SELECT COUNT(*) FROM products WHERE image_url IS NOT NULL AND image_url != ''",
            }
            for label, query in queries.items():
                self.cursor.execute(query)
                count = self.cursor.fetchone()[0]
                logger.info("   %s: %s", label, f"{count:,}")
        except Exception as exc:
            logger.error("Failed to get table stats: %s", exc)


# ── Main Entry Point ────────────────────────────────────


def main() -> None:
    """Run the metadata streaming pipeline."""
    db_config = DatabaseConfig()
    streamer = MetaDataStreamer(db_config)

    if not streamer.connect_to_database():
        return

    try:
        streamer.create_table()

        meta_file = Path(__file__).parent / "dataset" / "meta_AMAZON_FASHION.json.gz"
        success = streamer.stream_and_insert(meta_file, max_rows=None)

        if success:
            streamer.get_table_stats()
        else:
            logger.error("Processing failed.")
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
    finally:
        streamer.close_connection()


if __name__ == "__main__":
    main()
