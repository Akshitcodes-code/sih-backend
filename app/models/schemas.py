"""
Pydantic schemas = the "contract" between your backend and whoever
builds the frontend. These define exactly what JSON shape goes in
and out of each endpoint. Share this file with your frontend teammate
and they know exactly what to send/expect without asking you.
"""

from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    video_id: str
    filename: str
    duration_sec: float
    num_frames_processed: int
    num_transcript_chunks: int
    status: str


class SearchQuery(BaseModel):
    query: str
    top_k: int = 5


class SearchResultItem(BaseModel):
    video_id: str
    filename: str
    timestamp_sec: float
    matched_text: str
    source_type: str  # "transcript" (speech) or "ocr" (on-screen text)
    similarity_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    ai_summary: Optional[str] = None
