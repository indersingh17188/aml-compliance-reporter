from rag.retriever import AMLKnowledgeRetriever

if __name__ == "__main__":
    retriever = AMLKnowledgeRetriever()
    query = """High risk AML transaction with cross-currency movement, cross-bank transfer, unusual night-time activity, and large amount difference. Retrieve guidance for layering, currency conversion, and enhanced due diligence."""
    results = retriever.retrieve(query, top_k=5)
    print("\nRetrieved AML guidance:\n")
    for result in results:
        print("=" * 80)
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"][:700])
        print()
