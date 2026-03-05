import gzip
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import execute_values
from database_config import DatabaseConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stream_meta_to_postgres.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MetaDataStreamer:
    """Stream processor for meta_AMAZON_FASHION.json.gz data"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.connection = None
        self.cursor = None
        self.total_processed = 0
        self.total_inserted = 0
        self.total_skipped = 0
        self.batch_size = db_config.BATCH_SIZE
        
    def connect_to_database(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config.get_connection_params())
            self.cursor = self.connection.cursor()
            logger.info("Connected to PostgreSQL database successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def create_table(self):
        """Create products table if it doesn't exist"""
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
            logger.info("Products table created successfully")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            self.connection.rollback()
            raise
    
    def extract_product_data(self, json_line: str) -> Optional[Dict]:
        """Extract required fields from JSON line"""
        try:
            data = json.loads(json_line.strip())
            
            # Extract required fields
            asin = data.get('asin')
            title = data.get('title', '')
            brand = data.get('brand', '')
            image_url = data.get('imageURL', '')
            
            # Extract description from various possible fields
            description = self._extract_description(data)
            
            # Skip if ASIN is missing (required field)
            if not asin:
                return None
            
            return {
                'asin': asin,
                'title': title,
                'brand': brand,
                'image_url': image_url,
                'description': description
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON line: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error processing line: {e}")
            return None
    
    def _extract_description(self, data: Dict) -> str:
        """Extract description from various possible fields"""
        # Try different fields that might contain description
        description_fields = ['description', 'feature', 'details', 'product_description']
        
        for field in description_fields:
            if field in data:
                value = data[field]
                if isinstance(value, list):
                    # Join list items
                    return ' '.join(str(item) for item in value if item)
                elif isinstance(value, str):
                    return value
                elif value:
                    return str(value)
        
        # Try to extract from 'title' if it's too long (sometimes description is in title)
        title = data.get('title', '')
        if len(title) > 200:
            return title
        
        return ''
    
    def insert_batch(self, batch_data: List[Tuple]) -> bool:
        """Insert a batch of data into the database"""
        if not batch_data:
            return True
        
        insert_query = """
        INSERT INTO products (asin, title, brand, image_url, description)
        VALUES %s
        ON CONFLICT (asin) DO NOTHING
        """
        
        try:
            execute_values(
                self.cursor,
                insert_query,
                batch_data,
                template=None,
                page_size=len(batch_data)
            )
            self.connection.commit()
            self.total_inserted += len(batch_data)
            return True
        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            self.connection.rollback()
            return False
    
    def stream_and_insert(self, file_path: str, max_rows: Optional[int] = None):
        """Stream data from gzipped file and insert into database"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        logger.info(f"Starting to process: {file_path}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Max rows: {max_rows if max_rows else 'All'}")
        
        start_time = time.time()
        batch_data = []
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Check if we've reached the limit
                    if max_rows and self.total_processed >= max_rows:
                        break
                    
                    # Extract product data
                    product_data = self.extract_product_data(line)
                    
                    if product_data:
                        # Convert to tuple for bulk insert
                        row = (
                            product_data['asin'],
                            product_data['title'],
                            product_data['brand'],
                            product_data['image_url'],
                            product_data['description']
                        )
                        batch_data.append(row)
                        
                        # Insert when batch is full
                        if len(batch_data) >= self.batch_size:
                            if self.insert_batch(batch_data):
                                self.total_processed += len(batch_data)
                                logger.info(f"Processed: {self.total_processed}, "
                                          f"Inserted: {self.total_inserted}, "
                                          f"Skipped: {self.total_skipped}")
                            else:
                                self.total_skipped += len(batch_data)
                            
                            batch_data = []
                    else:
                        self.total_skipped += 1
                    
                    # Progress logging every 10,000 records
                    if line_num % 10000 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"Progress: {line_num:,} lines processed, "
                                  f"{elapsed:.2f}s elapsed")
        
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            return False
        
        # Insert remaining data in final batch
        if batch_data:
            if self.insert_batch(batch_data):
                self.total_processed += len(batch_data)
                logger.info(f"Final batch - Processed: {self.total_processed}, "
                          f"Inserted: {self.total_inserted}")
            else:
                self.total_skipped += len(batch_data)
        
        # Final statistics
        elapsed = time.time() - start_time
        logger.info(f"Processing completed!")
        logger.info(f"Total lines processed: {self.total_processed:,}")
        logger.info(f"Total records inserted: {self.total_inserted:,}")
        logger.info(f"Total records skipped: {self.total_skipped:,}")
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Average speed: {self.total_processed / elapsed:.2f} lines/second")
        
        return True
    
    def get_table_stats(self):
        """Get statistics about the products table"""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM products")
            total_records = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM products WHERE title IS NOT NULL AND title != ''")
            records_with_title = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM products WHERE brand IS NOT NULL AND brand != ''")
            records_with_brand = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM products WHERE image_url IS NOT NULL AND image_url != ''")
            records_with_image = self.cursor.fetchone()[0]
            
            logger.info(f"Database Statistics:")
            logger.info(f"   Total records: {total_records:,}")
            logger.info(f"   Records with title: {records_with_title:,}")
            logger.info(f"   Records with brand: {records_with_brand:,}")
            logger.info(f"   Records with image: {records_with_image:,}")
            
        except Exception as e:
            logger.error(f"Failed to get table stats: {e}")
    
    def close_connection(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")


def main():
    """Main execution function"""
    # Database configuration
    db_config = DatabaseConfig()
    
    # Initialize streamer
    streamer = MetaDataStreamer(db_config)
    
    # Connect to database
    if not streamer.connect_to_database():
        return
    
    try:
        # Create table
        streamer.create_table()
        
        # Stream data from meta file
        meta_file = "D:/MyWorkspace/recommendation-system/ai-model/data/dataset/meta_AMAZON_FASHION.json.gz"
        success = streamer.stream_and_insert(meta_file, max_rows=None)
        
        if success:
            # Show final statistics
            streamer.get_table_stats()
        else:
            logger.error("Processing failed")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    finally:
        # Always close connection
        streamer.close_connection()


if __name__ == "__main__":
    main()