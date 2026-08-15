"""
Step 1 of the pipeline: turn a raw video file into two things we can
actually run AI on:
  1. An audio track (.wav)  -> feeds the transcription service (ASR)
  2. A handful of sampled frames (.jpg) -> feeds the OCR service

We don't process every single frame — for a 30-60s Reel/Short, sampling
every N seconds (see config.frame_sample_interval_sec) is plenty and
keeps compute/cost down.
"""

import os
import cv2
from moviepy.editor import VideoFileClip

from app.config import settings


def extract_audio(video_path: str, video_id: str) -> str:
    """Pulls the audio track out of the video and saves it as a .wav file.
    Returns the path to the audio file."""
    audio_path = os.path.join(settings.video_dir, f"{video_id}.wav")
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        # Some short-form clips have no audio track at all — that's fine,
        # the search just won't have transcript results for this video.
        clip.close()
        return ""
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()
    return audio_path


def extract_frames(video_path: str, video_id: str) -> list[dict]:
    """Samples frames at a fixed interval and saves them as jpgs.
    Returns a list of {frame_path, timestamp_sec} dicts."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps * settings.frame_sample_interval_sec)

    frames_info = []
    frame_count = 0

    video_frame_dir = os.path.join(settings.frame_dir, video_id)
    os.makedirs(video_frame_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps
            frame_filename = f"frame_{frame_count}.jpg"
            frame_path = os.path.join(video_frame_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frames_info.append({
                "frame_path": frame_path,
                "timestamp_sec": round(timestamp_sec, 2),
            })

        frame_count += 1

    cap.release()
    return frames_info


def get_video_duration(video_path: str) -> float:
    clip = VideoFileClip(video_path)
    duration = clip.duration
    clip.close()
    return round(duration, 2)
