"""
Central place for all settings.
Everything is pulled from environment variables (see .env.example).
Using pydantic-settings means: if a required key is missing, the app
fails loudly at startup instead of crashing randomly later mid-request.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str
    chroma_persist_dir: str = "./storage/chroma"

    # Where uploaded videos and extracted frames live on disk
    video_dir: str = "./storage/videos"
    frame_dir: str = "./storage/frames"

    # How many frames per second to sample for OCR/object-detection.
    # 1 frame every 2 seconds is usually enough for short-form video
    # and keeps OCR/embedding costs sane.
    frame_sample_interval_sec: int = 2

    class Config:
        env_file = ".env"


settings = Settings()
