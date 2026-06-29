from typing import List, Dict


def chunk_documents(documents: List[Dict], chunk_size: int = 900, overlap: int = 150) -> List[Dict]:
    chunks = []
    for doc in documents:
        text, source = doc["text"], doc["source"]
        start, chunk_id = 0, 0
        while start < len(text):
            chunk_text = text[start:start + chunk_size].strip()
            if chunk_text:
                chunks.append({"source": source, "chunk_id": chunk_id, "text": chunk_text})
            chunk_id += 1
            start += chunk_size - overlap
    return chunks
