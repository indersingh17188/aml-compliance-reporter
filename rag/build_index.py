import os
import faiss
import joblib
from rag.knowledge_loader import load_markdown_documents
from rag.chunker import chunk_documents
from rag.embedding_model import embed_texts


def build_faiss_index(knowledge_base_dir: str = "knowledge_base", output_dir: str = "rag_index"):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading markdown documents...")
    documents = load_markdown_documents(knowledge_base_dir)
    print(f"Loaded {len(documents)} documents.")
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    texts = [chunk["text"] for chunk in chunks]
    print("Creating embeddings...")
    embeddings = embed_texts(texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, "aml_knowledge.faiss"))
    joblib.dump(chunks, os.path.join(output_dir, "chunks_metadata.joblib"))
    print(f"Saved FAISS index to: {output_dir}/aml_knowledge.faiss")
    print(f"Saved metadata to: {output_dir}/chunks_metadata.joblib")

if __name__ == "__main__":
    build_faiss_index()
