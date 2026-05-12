import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


def rerank_documents(question, retrieved_docs, top_n=2):
    if not GEMINI_API_KEY:
        return retrieved_docs[:top_n]

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    scored_docs = []

    for doc in retrieved_docs:
        prompt = f"""
You are a retrieval re-ranker.

Rate how relevant the following document chunk is for answering the user's question.

User question:
{question}

Document chunk:
{doc['content']}

Return only a number between 0 and 10.
"""

        try:
            response = model.generate_content(prompt)
            score_text = response.text.strip()
            score = float(score_text)
        except Exception:
            score = 0.0

        doc_copy = doc.copy()
        doc_copy["rerank_score"] = score
        scored_docs.append(doc_copy)

    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_docs[:top_n]