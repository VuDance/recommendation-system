from sentence_transformers import SentenceTransformer
import torch

_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _model

def encode_metadata(metadata):
    model = get_model()
    return model.encode(metadata, show_progress_bar=True)
