"""Database configuration for PostgreSQL connection."""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig:
    """Centralized database configuration.

    All connection parameters are read from environment variables
    with sensible defaults for local development.
    """

    # Connection parameters
    HOST: str = os.getenv("DB_HOST", "localhost")
    PORT: int = int(os.getenv("DB_PORT", "5432"))
    DATABASE: str = os.getenv("DB_NAME", "recommendation_db")
    USER: str = os.getenv("DB_USER", "postgres")
    PASSWORD: str = os.getenv("DB_PASSWORD", "password123")

    # Connection pool settings
    MIN_CONN: int = int(os.getenv("DB_MIN_CONN", "1"))
    MAX_CONN: int = int(os.getenv("DB_MAX_CONN", "20"))

    # Bulk insert settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1000"))

    @classmethod
    def get_connection_params(cls) -> dict[str, Any]:
        """Return connection parameters as a dictionary for psycopg2."""
        return {
            "host": cls.HOST,
            "port": cls.PORT,
            "database": cls.DATABASE,
            "user": cls.USER,
            "password": cls.PASSWORD,
        }

    @classmethod
    def get_connection_string(cls) -> str:
        """Return a psycopg2-compatible connection string."""
        return (
            f"postgresql://{cls.USER}:{cls.PASSWORD}"
            f"@{cls.HOST}:{cls.PORT}/{cls.DATABASE}"
        )
