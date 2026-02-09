import gc
import pandas as pd
from pymilvus import Collection, connections

def connect_milvus():
    try:
        connections.connect(alias="recommendationsystem", host="localhost", port="19530",password="Milvus",user="root", db_name="recommendationsystem")
        print("✅ Đã kết nối tới Milvus (alias: recommendationsystem).")
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")

def stream_dataframe_to_milvus(df, model, collection_name, batch_size=1000):
    # Khởi tạo collection (nó sẽ tự tìm alias="recommendationsystem")
    collection = Collection(collection_name, using='recommendationsystem')
    total_rows = len(df)
    print(f"Bắt đầu stream {total_rows} dòng vào Milvus...")

    for i in range(0, total_rows, batch_size):
        chunk = df.iloc[i : i + batch_size].copy()
        
        # Đảm bảo cột ID là cột đầu tiên và cột text đúng tên
        ids = chunk.iloc[:, 0].tolist() 
        texts = chunk['combined_info'].tolist()
        
        vectors = model.encode(texts, batch_size=256, show_progress_bar=False)
        
        collection.insert([ids, vectors.tolist()])
        
        del chunk, texts, vectors
        gc.collect()
        
        print(f"--- Đã nạp được {min(i + batch_size, total_rows)}/{total_rows} dòng ---")

    collection.flush()
    print("🚀 Hoàn thành nạp dữ liệu!")