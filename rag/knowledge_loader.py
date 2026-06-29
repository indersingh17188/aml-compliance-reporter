from pathlib import Path
from typing import List, Dict


def load_markdown_documents(knowledge_base_dir: str = "knowledge_base") -> List[Dict]:
    kb_path = Path(knowledge_base_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {knowledge_base_dir}")
    documents = []
    for file_path in sorted(kb_path.glob("*.md")):
        documents.append({"source": file_path.name, "text": file_path.read_text(encoding="utf-8")})
    return documents
