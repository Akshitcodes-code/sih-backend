"""
POST /upload
This is the endpoint that runs the ENTIRE pipeline end to end:
video file in -> audio extracted -> transcribed (ASR) -> frames
extracted -> OCR'd -> everything embedded -> stored in the vector DB.

Frontend just needs to send a multipart form file upload here — it
doesn't need to know anything about Whisper, OCR, or embeddings.
"""

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.models.schemas import UploadResponse
from app.services import video_processing, transcription, ocr, embeddings, vectorstore

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
        raise HTTPException(400, "Unsupported file type. Use mp4/mov/avi/webm.")

    video_id = str(uuid.uuid4())
    os.makedirs(settings.video_dir, exist_ok=True)
    video_path = os.path.join(settings.video_dir, f"{video_id}_{file.filename}")

    # Save the uploaded file to disk
    with open(video_path, "wb") as f:
        f.write(await file.read())

    duration = video_processing.get_video_duration(video_path)

    # --- ASR pipeline ---
    audio_path = video_processing.extract_audio(video_path, video_id)
    transcript_chunks = transcription.transcribe_audio(audio_path)

    if transcript_chunks:
        transcript_embeddings = embeddings.embed_texts_batch(
            [c["text"] for c in transcript_chunks]
        )
        vectorstore.add_chunks(
            video_id, file.filename, transcript_chunks, "transcript", transcript_embeddings
        )

    # --- OCR pipeline ---
    frames_info = video_processing.extract_frames(video_path, video_id)
    ocr_chunks = ocr.process_frames_ocr(frames_info)

    if ocr_chunks:
        ocr_embeddings = embeddings.embed_texts_batch([c["text"] for c in ocr_chunks])
        vectorstore.add_chunks(video_id, file.filename, ocr_chunks, "ocr", ocr_embeddings)

    return UploadResponse(
        video_id=video_id,
        filename=file.filename,
        duration_sec=duration,
        num_frames_processed=len(frames_info),
        num_transcript_chunks=len(transcript_chunks),
        status="processed",
    )
