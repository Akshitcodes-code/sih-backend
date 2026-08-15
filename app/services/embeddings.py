"""
Step 4: turn text chunks (from ASR + OCR) into embedding vectors —
now using Gemini's free embedding model instead of OpenAI's.
"""

from google import genai
from app.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

EMBEDDING_MODEL = "gemini-embedding-001"


def embed_text(text: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return response.embeddings[0].values


def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """Batching is faster than calling embed_text() in a loop."""
    if not texts:
        return []
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in response.embeddings]
