import json
import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


def verify_answer(question, answer, retrieved_docs):
    if not GEMINI_API_KEY:
        return {
            "claims": [
                {
                    "claim": answer,
                    "support": "Unknown",
                    "reason": "GEMINI_API_KEY bulunamadı."
                }
            ],
            "overall_risk": "Unknown"
        }

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    context = "\n\n".join(
        [f"{doc['title']}: {doc['content']}" for doc in retrieved_docs]
    )

    prompt = f"""
You are a verification assistant.

Your task:
1. Read the question, context, and generated answer.
2. Break the answer into claims or sentences.
3. For each claim, decide whether it is:
   - Supported
   - Partially Supported
   - Unsupported
4. Give a short reason for each.
5. At the end, provide an overall risk label:
   - Low
   - Medium
   - High

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "claim": "text",
      "support": "Supported or Partially Supported or Unsupported",
      "reason": "short explanation"
    }}
  ],
  "overall_risk": "Low or Medium or High"
}}

Question:
{question}

Context:
{context}

Generated Answer:
{answer}
"""

    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    try:
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)
    except Exception:
        return {
            "claims": [
                {
                    "claim": answer,
                    "support": "Unknown",
                    "reason": f"Could not parse verifier output. Raw output: {raw_text}"
                }
            ],
            "overall_risk": "Unknown"
        }