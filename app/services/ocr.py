"""
Step 3: OCR (on-screen text) — captions, memes, and text overlays baked
into Reels/Shorts often carry meaning speech doesn't (e.g. a joke's
punchline shown as text, not said out loud). We run Tesseract OCR on
each sampled frame.

NOTE ON OBJECT DETECTION / VLM (extension point):
The full problem statement also asks for object detection and VLM-based
understanding (e.g. "what's happening in this scene", not just text).
That's intentionally left as a plug-in point here — this function is
where you'd add a call to something like a YOLO model (objects) or a
vision-language model like GPT-4o/CLIP (scene description). Doing that
requires either a GPU or paid vision API calls, so it's not wired in
by default — search for "EXTENSION POINT" below.
"""

import pytesseract
from PIL import Image


def extract_text_from_frame(frame_path: str) -> str:
    """Runs OCR on a single frame, returns cleaned text (may be empty)."""
    try:
        image = Image.open(frame_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:
        # A bad/corrupt frame shouldn't crash the whole upload pipeline
        return ""


def process_frames_ocr(frames_info: list[dict]) -> list[dict]:
    """Takes the frame list from video_processing.extract_frames and
    attaches OCR text to each one. Frames with no detected text are
    dropped since there's nothing to search on."""
    results = []
    for frame in frames_info:
        text = extract_text_from_frame(frame["frame_path"])
        if text:
            results.append({
                "text": text,
                "timestamp_sec": frame["timestamp_sec"],
            })

        # EXTENSION POINT: object detection / VLM scene captioning
        # e.g. objects = yolo_model(frame["frame_path"])
        #      caption = vlm_client.describe(frame["frame_path"])
        # then append those as additional searchable text chunks too.

    return results
