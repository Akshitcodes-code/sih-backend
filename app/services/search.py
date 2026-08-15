"""
Step 6: search + RAG.
Two things happen here:
 1. Semantic search: embed the user's query, find the closest chunks
    in the vector store (this alone answers "which videos/moments
    match this query").
 2. RAG (Retrieval-Augmented Generation): take those retrieved chunks
    and feed them to a chat model to produce a natural-language answer
    ("Retrieval-Augmented" = the model's answer is grounded in what we
    actually retrieved, not just its own memory).
"""

from google import genai
from app.config import settings
from app.services.embeddings import embed_text
from app.services import vectorstore

client = genai.Client(api_key=settings.gemini_api_key)
SUMMARY_MODEL = "gemini-3.5-flash-lite"  # fastest/cheapest, free-tier friendly


def semantic_search(query_text: str, top_k: int = 5) -> list[dict]:
    query_embedding = embed_text(query_text)
    raw = vectorstore.query(query_embedding, top_k=top_k)

    results = []
    # Chroma returns parallel lists wrapped in an extra outer list
    # (because you could technically batch multiple queries at once)
    docs = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        results.append({
            "video_id": meta["video_id"],
            "filename": meta["filename"],
            "timestamp_sec": meta["timestamp_sec"],
            "matched_text": doc,
            "source_type": meta["source_type"],
            # Chroma gives distance (lower = closer); flip to a similarity-style score
            "similarity_score": round(1 - dist, 4),
        })
    return results


def generate_rag_summary(query_text: str, results: list[dict]) -> str:
    """Optional: ask the model to summarize what it found in plain
    language, grounded ONLY in the retrieved chunks (not its own
    knowledge) — this is the 'RAG' part of the problem statement."""
    if not results:
        return "No matching content found."

    context = "\n".join(
        f"- [{r['filename']} @ {r['timestamp_sec']}s, {r['source_type']}]: {r['matched_text']}"
        for r in results
    )

    prompt = (
        f"A user searched for: \"{query_text}\"\n\n"
        f"Here are the matching video moments retrieved from the database:\n{context}\n\n"
        "In 2-3 sentences, summarize what was found and why it's relevant "
        "to the search query. Only use the information above, don't invent anything."
    )

    response = client.models.generate_content(
        model=SUMMARY_MODEL,
        contents=prompt,
    )
    return response.text.strip()
