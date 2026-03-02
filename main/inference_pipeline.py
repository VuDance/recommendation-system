import pandas as pd
import torch
from pymilvus import Collection
import logging
from util.util import load_encoders, connect_to_milvus, load_user_tower, get_user_history, get_user_vector, query_milvus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ✅ FIX: khởi tạo resources 1 lần duy nhất, không load lại mỗi request
class RecommendationPipeline:
    def __init__(self):
        logger.info("Initializing pipeline (once)...")

        self.product_encoder, self.brand_encoder = load_encoders()
        self.user_features_df = pd.read_parquet('processed_data/user_features.parquet')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_tower = load_user_tower(
            'model/user_tower_checkpoint.pth',
            # ✅ FIX: lấy vocab size từ encoder
            product_vocab_size=len(self.product_encoder.classes_)
        ).to(self.device)

        if not connect_to_milvus():
            raise RuntimeError("Cannot connect to Milvus")

        # ✅ FIX: load collection vào memory 1 lần
        self.collection = Collection(name="product_vectors")
        self.collection.load()
        logger.info("Pipeline ready!")

    def recommend(self, user_id, k=10):
        history = get_user_history(user_id, self.user_features_df)
        if history is None:
            return []

        user_vector = get_user_vector(history, self.user_tower, self.device)
        return query_milvus(user_vector, self.collection, k=k)


def main():
    pipeline = RecommendationPipeline()

    user_features_df = pd.read_parquet('processed_data/user_features.parquet')
    sample_user = user_features_df.iloc[0]['reviewerID']

    print(f"\n=== TEST RECOMMENDATION ===")
    print(f"User: {sample_user}")

    recommendations = pipeline.recommend(sample_user, k=10)

    for rec in recommendations:
        print(f"{rec['rank']:2d}. {rec['product_id']}  score: {rec['score']:.4f}")


if __name__ == "__main__":
    main()