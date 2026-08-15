"""
Entry point. Run with:  uvicorn app.main:app --reload

CORS is enabled wide-open here (allow_origins=["*"]) so your teammate's
plain HTML/CSS/JS frontend can call this API directly from the browser
during development. Tighten this to your actual frontend's domain
before you ever deploy this for real.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import upload, search

app = FastAPI(
    title="Reel/Shorts Semantic Search API",
    description="SIH PS14 backend: upload a video, search it semantically.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, tags=["upload"])
app.include_router(search.router, tags=["search"])


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend is running"}
