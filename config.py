import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st

    GEMINI_API_KEY = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY")
    )

    MODEL_NAME = st.secrets.get(
        "MODEL_NAME",
        os.getenv("MODEL_NAME", "gemini-2.5-flash")
    )

except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
