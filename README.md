# Reel/Shorts Semantic Search — Backend (PS14 scaffold)

A working FastAPI backend for the SIH problem statement: upload a
short-form video, and later search it using natural language ("dog
jumping into pool") instead of keywords.

## How to run it

```bash
cd sih-backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then paste your real Gemini key into .env
# Get a free key (no credit card needed) at https://aistudio.google.com/apikey

# You also need Tesseract OCR installed on your system (not a pip package):
#   Ubuntu/Debian: sudo apt install tesseract-ocr
#   Mac:           brew install tesseract
#   Windows:       https://github.com/UB-Mannheim/tesseract/wiki

uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` — FastAPI auto-generates an
interactive API tester (Swagger UI). Upload a short video there
first, then hit `/search` with a text query. This is the fastest
way to see the whole thing work without writing a single frontend
line yet.

## The pipeline, in order (this is your reverse-engineering map)

```
video file
   │
   ├─► video_processing.py  → splits into audio (.wav) + sampled frames (.jpg)
   │
   ├─► transcription.py     → audio → Gemini → timestamped transcript (ASR)
   │
   ├─► ocr.py                → frames → Tesseract → on-screen text (OCR)
   │        (object detection / VLM scene captioning is a marked
   │         extension point inside this file — not wired in by default)
   │
   ├─► embeddings.py         → all text chunks → Gemini embedding vectors (free tier)
   │
   └─► vectorstore.py        → vectors stored in Chroma (local vector DB),
                                tagged with video_id + timestamp

search query
   │
   ├─► embeddings.py         → query → embedding vector
   ├─► vectorstore.py        → nearest-neighbor lookup → matching chunks
   └─► search.py (RAG part)  → matching chunks → Gemini → plain-English summary
```

`routes/upload.py` and `routes/search.py` are just thin wrappers that
call the above in order — read those two files first to see the
whole flow, THEN dig into each service file.

## Suggested order to actually study this

1. `app/main.py` — see how the app is wired together, what CORS is doing
2. `app/routes/upload.py` — the full ingestion flow, step by step
3. `app/services/video_processing.py` — simplest service, good warm-up
4. `app/services/transcription.py` + `ocr.py` — the two "understanding" steps
5. `app/services/embeddings.py` + `vectorstore.py` — how semantic search actually works
6. `app/services/search.py` — where retrieval becomes RAG
7. `app/routes/search.py` — how it all comes back together

## What's intentionally NOT built in (so you know what's missing, not broken)

- **Object detection** (e.g. YOLO) — flagged with `EXTENSION POINT` comment in `ocr.py`
- **VLM scene understanding** (e.g. GPT-4o vision, CLIP) — same extension point
- **Auth / user accounts** — not needed for a hackathon demo
- **Async/background job queue for uploads** — right now upload is
  synchronous (frontend waits while it processes). Fine for a demo;
  for real use you'd move this to a background task/queue so upload
  returns instantly.

## Connecting your teammate's HTML/CSS/JS frontend

They just need two `fetch()` calls:
- `POST /upload` — multipart form, field name `file`
- `POST /search` — JSON body `{"query": "...", "top_k": 5}`

Full request/response shapes are in `app/models/schemas.py` — send
them that file, they don't need to read anything else.
