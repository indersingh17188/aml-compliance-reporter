from sentence_transformers import SentenceTransformer

_MODEL = None

def load_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL

def embed_texts(texts, model_name: str = "BAAI/bge-small-en-v1.5"):
    model = load_embedding_model(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
