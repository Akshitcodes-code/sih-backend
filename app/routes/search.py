"""
POST /search
Frontend sends {"query": "cat falling off table", "top_k": 5} and gets
back matching video moments plus an optional plain-English summary.
"""

from fastapi import APIRouter
from app.models.schemas import SearchQuery, SearchResponse, SearchResultItem
from app.services import search as search_service

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_videos(payload: SearchQuery):
    results = search_service.semantic_search(payload.query, top_k=payload.top_k)
    summary = search_service.generate_rag_summary(payload.query, results)

    return SearchResponse(
        query=payload.query,
        results=[SearchResultItem(**r) for r in results],
        ai_summary=summary,
    )
