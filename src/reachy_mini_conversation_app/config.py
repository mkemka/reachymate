import os
import logging

from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

# Locate .env file (search upward from current working directory)
dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    # Load .env and override environment variables
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Configuration loaded from {dotenv_path}")
else:
    logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is downloaded in console.py if needed

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

    # Receptionist / biometric gate (optional; use with --receptionist and pip install '.[receptionist]')
    RECEPTIONIST_MODE = os.getenv("RECEPTIONIST_MODE", "").strip().lower() in ("1", "true", "yes")
    RECEPTIONIST_FACE_THRESHOLD = float(os.getenv("RECEPTIONIST_FACE_THRESHOLD", "0.35"))
    # Min Whisper phrase similarity (difflib) vs enrolled passphrase (see openai-whisper)
    RECEPTIONIST_VOICE_THRESHOLD = float(os.getenv("RECEPTIONIST_VOICE_THRESHOLD", "0.72"))
    RECEPTIONIST_SESSION_TTL_S = float(os.getenv("RECEPTIONIST_SESSION_TTL_S", "300"))
    RECEPTIONIST_AUDIO_BUFFER_S = float(os.getenv("RECEPTIONIST_AUDIO_BUFFER_S", "6"))
    # Comma-separated substrings; user must say one after face+voice unlock before the robot speaks
    RECEPTIONIST_WAKE_PHRASES = os.getenv(
        "RECEPTIONIST_WAKE_PHRASES",
        "hey reachy,hello reachy,reachy",
    )
    # Face detection: Ultralytics YOLO26x (highest accuracy, ~119 MB)
    # https://platform.ultralytics.com/ultralytics/yolo26/yolo26x
    # Options: yolo26x.pt (best), yolo26n.pt (fastest), "hf" (HF face-specific detector)
    YOLO_FACE_MODEL = os.getenv("YOLO_FACE_MODEL", "yolo26x.pt").strip()
    YOLO_FACE_MODEL_REPO = os.getenv("YOLO_FACE_MODEL_REPO", "AdamCodd/YOLOv11n-face-detection")
    YOLO_FACE_MODEL_FILENAME = os.getenv("YOLO_FACE_MODEL_FILENAME", "model.pt")
    # Face embedding: InsightFace buffalo_l (best open-source recognition model)
    # https://www.insightface.ai/ — install with: pip install '.[receptionist]'
    INSIGHTFACE_MODEL_NAME = os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l")
    # Voice recognition: OpenAI Whisper — https://github.com/openai/whisper
    # Options: tiny.en, base.en (default), small.en, medium.en, turbo, large
    # English-only (.en) variants are faster; turbo is best speed/accuracy if GPU available
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en").strip()

    # Optional: require this exact phrase (case-insensitive) before mic audio is
    # sent to the model and before assistant audio plays. Unlock via dashboard
    # chat, POST /activation/unlock, or typing the phrase in the chat box.
    REACHY_ACTIVATION_PHRASE = os.getenv("REACHY_ACTIVATION_PHRASE", "").strip() or None

    # ── Ledger & Check-In Controller ──────────────────────────────────────
    # Points deducted per successful check-in (default 1)
    CHECK_IN_COST_POINTS = int(os.getenv("CHECK_IN_COST_POINTS", "1"))
    # Points given to a newly enrolled member
    LEDGER_INITIAL_BALANCE = int(os.getenv("LEDGER_INITIAL_BALANCE", "100"))
    # Face cosine similarity score that triggers the HIGH-CONFIDENCE path (no voice check needed)
    FACE_CONFIDENCE_HIGH_THRESHOLD = float(os.getenv("FACE_CONFIDENCE_HIGH_THRESHOLD", "0.90"))


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass
