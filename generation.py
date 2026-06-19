import os
from typing import List

from dotenv import load_dotenv
from google import genai

load_dotenv()

ANSWER_PROMPT = """You are a financial and aviation analyst.

Answer the question using ONLY the provided context.

If the context does not contain enough information to answer the question, explicitly state that the answer cannot be determined from the corpus.

Provide source citations in your answer using the document name and page number from the context labels.

Question:
{question}

Context:
{context}
"""


def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return api_key


def get_gemini_model(model: str | None = None) -> str:
    return model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def generate_answer(question: str, context: str, model: str | None = None) -> str:
    if not context.strip():
        return "Insufficient evidence found in the corpus."

    client = genai.Client(api_key=get_gemini_api_key())
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    response = client.models.generate_content(
        model=get_gemini_model(model),
        contents=prompt,
    )
    return response.text.strip()


def format_sources_section(citations: List[str]) -> str:
    if not citations:
        return "Sources:\n- None"
    lines = "\n".join(f"- {citation}" for citation in citations)
    return f"Sources:\n{lines}"
