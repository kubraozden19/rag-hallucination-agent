import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


def reformulate_query(user_question):
    if not GEMINI_API_KEY:
        return user_question

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    prompt = f"""
You are a query reformulation assistant.

Your task:
- Rewrite the user's question into a short, clear search query for retrieval.
- Keep the meaning exactly the same.
- If the user question is in Turkish, rewrite it in English for better retrieval.
- Do not answer the question.
- Return only the reformulated query text.

User question:
{user_question}
"""

    response = model.generate_content(prompt)
    return response.text.strip()