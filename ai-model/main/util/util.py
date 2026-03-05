import logging
import pickle
import os
from dotenv import load_dotenv
from pymilvus import connections
import torch
from user_tower import UserTower

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_encoders():
    logger.info("Loading encoders...")
    with open('processed_data/product_encoder.pkl', 'rb') as f:
        product_encoder = pickle.load(f)
    with open('processed_data/brand_encoder.pkl', 'rb') as f:
        brand_encoder = pickle.load(f)
    return product_encoder, brand_encoder

def connect_to_milvus():
    try:
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
            user=os.getenv("MILVUS_USER", "root"),
            password=os.getenv("MILVUS_PASSWORD")
        )
        logger.info("Connected to Milvus!")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False


def load_user_tower(checkpoint_path, product_vocab_size):
    logger.info(f"Loading UserTower from {checkpoint_path}")
    # ✅ FIX: dùng product_vocab_size từ encoder thay vì hardcode
    user_tower = UserTower(product_vocab_size=product_vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    user_tower.load_state_dict(checkpoint['model_state_dict'])
    user_tower.eval()
    return user_tower


def get_user_history(user_id, user_features_df):
    user_row = user_features_df[user_features_df['reviewerID'] == user_id]
    if user_row.empty:
        logger.warning(f"User {user_id} not found")
        return None

    history = user_row['history_padded'].iloc[0]

    # Bỏ padding (0), giữ lại index thực
    actual_history = [p for p in history if p != 0]

    if len(actual_history) < 3:
        logger.warning(f"History too short: {len(actual_history)} items")
        return None

    return actual_history  # đã là index, không cần encode lại


def get_user_vector(user_history, user_tower, device):
    history_tensor = torch.tensor(user_history, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
    with torch.no_grad():
        user_vector = user_tower(history_tensor.to(device))
    return user_vector.cpu().numpy().flatten()  # (64,)


def query_milvus(user_vector, collection, k=10):
    results = collection.search(
        data=[user_vector.tolist()],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=k,
        output_fields=["product_id"]
    )

    recommendations = []
    for hits in results:
        for rank, hit in enumerate(hits, 1):  # ✅ FIX: dùng enumerate thay hit.rank
            recommendations.append({
                'product_id': hit.entity.get('product_id'),
                'score'     : hit.distance,
                'rank'      : rank
            })
    return recommendations

