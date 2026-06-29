# RAG Starter Pack Instructions

Copy these folders/files into your project root:

- `knowledge_base/`
- `rag/`
- `requirements_rag.txt`

Install dependencies:

```bash
pip install -r requirements_rag.txt
```

Build the FAISS index:

```bash
python -m rag.build_index
```

Test retrieval:

```bash
python -m rag.test_retriever
```

Expected sources may include layering, cross-border transfers, unusual transaction timing, currency conversion patterns, and enhanced due diligence.

Next integration step: add the retriever into `app.py` after SHAP explanation is generated.
