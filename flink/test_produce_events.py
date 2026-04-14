"""
Test script: sends real product view events to Kafka topic 'user-view-events'.
Run after docker-compose is up and Kafka is ready.
"""
import json
import time
from kafka import KafkaProducer

BOOTSTRAP_SERVERS = ["localhost:9092"]
TOPIC = "user-view-events"

# Real products from ai-model/data/dataset/products_clean.csv
REAL_PRODUCTS = [
    "1",  # Slime Time Fall Fest
    "2",  # XCC Qi promise new spider snake
    "3",  # Magical Things I Really Do Do Too!
    "4",  # Ashes to Ashes, Oranges to Oranges
    "5",  # Aether & Empire #1
    "1",  # 365 Affirmations for a Year of Love
    "2",  # Blessed by Pope Benedetto XVI Bracelet
    "2",  # Womens Sexy Sleeveless Camouflage Print
    "1",  # Sevendayz Men's Shady Records Eminem Hoodie
    "4",  # Dante's Peak - Laserdisc
    "3",  # Milliongadgets Earring Safety Backs
    "5",  # Envirosax Kids Series Jessie & Lulu
    "8",  # Envirosax Greengrocer Series Bag
    "34",  # Blessed by Pope Benedetto XVI Rosary
    "13",  # Tideclothes ALAGIRLS Strapless Beading
]

TEST_USERS = ["user-001", "user-002", "user-003", "user-004", "user-005"]


def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        api_version=(3, 6, 0),
        request_timeout_ms=10000,
    )

    print(f"Sending events to topic '{TOPIC}'...")

    for user_id in TEST_USERS:
        for i in range(3):
            event = {
                "userId": user_id,
                "productId": REAL_PRODUCTS[(hash(user_id) + i) % len(REAL_PRODUCTS)],
                "timestamp": int(time.time() * 1000),
            }
            producer.send(TOPIC, value=event)
            print(f"  Sent: {event}")

    # Flush and close
    producer.flush()
    producer.close()
    print(f"\nDone! {len(TEST_USERS) * 3} events sent.")


if __name__ == "__main__":
    main()
