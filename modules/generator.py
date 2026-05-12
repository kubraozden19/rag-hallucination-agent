import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


def generate_answer(question, retrieved_docs):
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY bulunamadı. Lütfen .env dosyasını kontrol et."

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    context = "\n\n".join(
        [f"{doc['title']}: {doc['content']}" for doc in retrieved_docs]
    )

    prompt = f"""
You are a helpful assistant.
Answer the user's question using only the provided context.
If the answer is not clearly supported by the context, say that the information is not available in the provided sources.

Question:
{question}

Context:
{context}

Answer:
"""

    response = model.generate_content(prompt)
    return response.text.strip()