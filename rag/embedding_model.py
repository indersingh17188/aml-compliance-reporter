import numpy as np
from fastembed import TextEmbedding


_MODEL = None


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a lightweight local embedding model using FastEmbed.

    FastEmbed uses ONNX Runtime instead of PyTorch, which makes it more stable
    for local CPU deployment and Streamlit apps.
    """
    global _MODEL

    if _MODEL is None:
        _MODEL = TextEmbedding(model_name=model_name)

    return _MODEL


def embed_texts(texts, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Convert texts into normalized dense embeddings for FAISS similarity search.
    """
    model = load_embedding_model(model_name)

    embeddings = list(model.embed(texts))
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize embeddings so FAISS inner product behaves like cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    return embeddings