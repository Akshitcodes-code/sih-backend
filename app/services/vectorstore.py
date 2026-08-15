"""
Step 5: the vector database. This is where all the embedded chunks
(transcript + OCR text, tagged with video_id/timestamp) get stored,
and where semantic search actually happens (nearest-neighbor lookup
over vectors).

We use Chroma because it runs entirely locally (no separate server,
no extra API key) — good for a hackathon where you don't want another
moving piece to deploy. Swappable later for Pinecone/Weaviate/Qdrant
if you need cloud-hosted scale.
"""

import chromadb
from app.config import settings

_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
_collection = _client.get_or_create_collection(name="video_chunks")


def add_chunks(
    video_id: str,
    filename: str,
    chunks: list[dict],
    source_type: str,
    embeddings: list[list[float]],
) -> None:
    """chunks: list of {text, timestamp_sec / start_sec} dicts.
    source_type: 'transcript' or 'ocr'."""
    if not chunks:
        return

    ids = [f"{video_id}_{source_type}_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "video_id": video_id,
            "filename": filename,
            "timestamp_sec": c.get("timestamp_sec", c.get("start_sec", 0)),
            "source_type": source_type,
        }
        for c in chunks
    ]

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(query_embedding: list[float], top_k: int = 5) -> dict:
    return _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
