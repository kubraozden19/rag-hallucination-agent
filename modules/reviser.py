import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


def revise_answer(question, original_answer, verification, retrieved_docs):
    if not GEMINI_API_KEY:
        return original_answer

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    context = "\n\n".join(
        [f"{doc['title']}: {doc['content']}" for doc in retrieved_docs]
    )

    claims_text = ""
    for item in verification.get("claims", []):
        claims_text += (
            f"Claim: {item.get('claim', '')}\n"
            f"Support: {item.get('support', '')}\n"
            f"Reason: {item.get('reason', '')}\n\n"
        )

    prompt = f"""
You are an answer reviser.

Your task:
- Rewrite the original answer so that it is safer, more cautious, and fully aligned with the provided context.
- If any claim is unsupported or partially supported, avoid stating it as a definite fact.
- Use only the provided context.
- Keep the answer short and clear.
- Return only the revised final answer.

Question:
{question}

Context:
{context}

Original Answer:
{original_answer}

Verification Results:
{claims_text}
"""

    response = model.generate_content(prompt)
    return response.text.strip()