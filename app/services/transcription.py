"""
Step 2: ASR (Automatic Speech Recognition) — now using Gemini instead
of Whisper. Gemini doesn't have a dedicated "transcription" endpoint
like Whisper does — instead we upload the audio file and PROMPT the
model to transcribe it with timestamps, forcing structured JSON output
so we get back the same {text, start_sec, end_sec} shape as before.
This means the rest of the pipeline (embeddings.py, vectorstore.py)
doesn't need to change at all.
"""

import time
from google import genai
from google.genai import types
from app.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

TRANSCRIPTION_MODEL = "gemini-3.5-flash"

# Force the model to return exactly this JSON shape, so we don't have
# to parse messy free-text output.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                },
                "required": ["text", "start_sec", "end_sec"],
            },
        }
    },
    "required": ["segments"],
}

_PROMPT = (
    "Transcribe the speech in this audio file. Break it into natural "
    "segments (a sentence or short phrase each). For each segment give "
    "the spoken text and its start/end time in seconds from the start "
    "of the audio. If there is no speech, return an empty segments list."
)


def transcribe_audio(audio_path: str) -> list[dict]:
    """Returns a list of {text, start_sec, end_sec} chunks.
    Empty list if there's no audio (silent video)."""
    if not audio_path:
        return []

    # Gemini's Files API: upload once, then reference it in the prompt.
    # Files are auto-deleted by Google after 48 hours, no cleanup needed.
    uploaded_file = client.files.upload(file=audio_path)

    # Uploaded audio needs a moment to finish processing on Google's side
    # before it can be used in a generate_content call.
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(1)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        return []

    response = client.models.generate_content(
        model=TRANSCRIPTION_MODEL,
        contents=[_PROMPT, uploaded_file],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        ),
    )

    import json
    data = json.loads(response.text)
    chunks = []
    for seg in data.get("segments", []):
        chunks.append({
            "text": seg["text"].strip(),
            "start_sec": round(float(seg["start_sec"]), 2),
            "end_sec": round(float(seg["end_sec"]), 2),
        })
    return chunks
