import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.utils import chunk_text


def load_documents(path="data/documents.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunks():
    docs = load_documents()
    chunked_docs = []

    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size=20, overlap=5)
        for i, chunk in enumerate(chunks, start=1):
            chunked_docs.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "chunk_id": i,
                    "content": chunk
                }
            )

    return chunked_docs


def retrieve_relevant_docs(query, top_k=3):
    docs = build_chunks()
    texts = [doc["content"] for doc in docs]

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        results.append(
            {
                "id": docs[idx]["id"],
                "title": docs[idx]["title"],
                "chunk_id": docs[idx]["chunk_id"],
                "content": docs[idx]["content"],
                "score": float(similarities[idx]),
            }
        )

    return results