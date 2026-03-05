"""
Database configuration for PostgreSQL connection
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseConfig:
    """Database configuration class"""
    
    # Database connection parameters
    HOST = os.getenv('DB_HOST', 'localhost')
    PORT = int(os.getenv('DB_PORT', 5432))
    DATABASE = os.getenv('DB_NAME', 'recommendation_db')
    USER = os.getenv('DB_USER', 'postgres')
    PASSWORD = os.getenv('DB_PASSWORD', 'password123')
    
    # Connection pool settings
    MIN_CONN = int(os.getenv('DB_MIN_CONN', 1))
    MAX_CONN = int(os.getenv('DB_MAX_CONN', 20))
    
    # Bulk insert settings
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1000))
    
    @classmethod
    def get_connection_params(cls):
        """Get connection parameters as dictionary"""
        return {
            'host': cls.HOST,
            'port': cls.PORT,
            'database': cls.DATABASE,
            'user': cls.USER,
            'password': cls.PASSWORD
        }
    
    @classmethod
    def get_connection_string(cls):
        """Get connection string for psycopg2"""
        return f"postgresql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.DATABASE}"