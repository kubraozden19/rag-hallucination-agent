import json
import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


def run_cloud_demo_agent(question, retrieved_docs):
    if not GEMINI_API_KEY:
        return {
            "answer": "GEMINI_API_KEY bulunamadı. Lütfen Streamlit Secrets ayarlarını kontrol edin.",
            "claims": [],
            "overall_risk": "Unknown"
        }

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    context = "\n\n".join(
        [f"{doc['title']}: {doc['content']}" for doc in retrieved_docs]
    )

    prompt = f"""
You are a RAG answer verification agent.

Your task:
1. Answer the user's question using only the provided context.
2. Then verify your own answer claim by claim.
3. For each claim, decide whether it is:
   - Supported
   - Partially Supported
   - Unsupported
4. Give a short reason for each claim.
5. Provide an overall risk label:
   - Low
   - Medium
   - High

Return ONLY valid JSON in this exact format:
{{
  "answer": "final answer text",
  "claims": [
    {{
      "claim": "claim text",
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
"""

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)

    except Exception:
        return {
            "answer": "Model çağrısı sırasında hata oluştu. Bu durum genellikle ücretsiz API kota sınırı veya geçici servis limiti nedeniyle oluşur.",
            "claims": [
                {
                    "claim": "Model response could not be generated.",
                    "support": "Unknown",
                    "reason": "The model call could not be completed because the API quota or service limit was exceeded."
                }
            ],
            "overall_risk": "Unknown"
        }
