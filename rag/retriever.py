import faiss
import joblib
from rag.embedding_model import embed_texts


class AMLKnowledgeRetriever:
    def __init__(self, index_path: str = "rag_index/aml_knowledge.faiss", metadata_path: str = "rag_index/chunks_metadata.joblib"):
        self.index = faiss.read_index(index_path)
        self.chunks = joblib.load(metadata_path)

    def retrieve(self, query: str, top_k: int = 4):
        query_embedding = embed_texts([query])
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({"score": float(score), "source": chunk["source"], "chunk_id": chunk["chunk_id"], "text": chunk["text"]})
        return results


def build_query_from_prediction(risk_level: str, shap_reasons, transaction_row=None) -> str:
    query_parts = [f"AML guidance for a {risk_level} transaction.", "Relevant suspicious activity patterns and analyst actions."]
    if shap_reasons:
        query_parts.append("Model explanation factors:")
        query_parts.extend([str(reason) for reason in shap_reasons])
    if transaction_row is not None:
        try:
            if "cross_currency" in transaction_row and int(transaction_row["cross_currency"]) == 1:
                query_parts.append("cross-currency transaction currency conversion")
            if "cross_bank_transfer" in transaction_row and int(transaction_row["cross_bank_transfer"]) == 1:
                query_parts.append("cross-bank transfer layering")
            if "is_night_transaction" in transaction_row and int(transaction_row["is_night_transaction"]) == 1:
                query_parts.append("night transaction unusual timing")
        except Exception:
            pass
    return "\n".join(query_parts)
