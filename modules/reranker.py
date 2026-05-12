def rerank_documents(question, retrieved_docs, top_n=2):
    """
    Local score-based reranker.

    This function does not call any external LLM API.
    It re-sorts retrieved candidate chunks according to their retrieval score
    and selects the top_n strongest evidence chunks.
    """

    scored_docs = []

    for doc in retrieved_docs:
        doc_copy = doc.copy()
        doc_copy["rerank_score"] = doc_copy.get("score", 0.0)
        scored_docs.append(doc_copy)

    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_docs[:top_n]
