"""Wire store, embedders, and gate for receptionist mode."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from pathlib import Path

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.receptionist.gate import ReceptionistGate
from reachy_mini_conversation_app.receptionist.store import EnrollmentStore
from reachy_mini_conversation_app.receptionist.face_embed import try_face_embedder
from reachy_mini_conversation_app.receptionist.whisper_voice import try_whisper_transcriber


if TYPE_CHECKING:
    from reachy_mini_conversation_app.camera_worker import CameraWorker
    from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker


logger = logging.getLogger(__name__)


def create_receptionist_stack(
    instance_path: str | Path | None,
    camera_worker: CameraWorker,
    head_tracker: HeadTracker | None,
) -> ReceptionistGate:
    """Build a :class:`ReceptionistGate` or raise with an actionable error."""
    root = Path(instance_path or Path.cwd()) / "receptionist_data"
    store = EnrollmentStore(root)

    face = try_face_embedder(getattr(config, "INSIGHTFACE_MODEL_NAME", "buffalo_l"))
    voice = try_whisper_transcriber(getattr(config, "WHISPER_MODEL", "base"), device=None)
    if face is None or voice is None:
        raise RuntimeError(
            "Receptionist dependencies missing. Install with: pip install '.[receptionist]' "
            "(InsightFace + ONNX Runtime + openai-whisper + PyTorch; Whisper needs ffmpeg on PATH). "
            "Docs: https://github.com/openai/whisper — https://www.insightface.ai/",
        )

    wake_raw = getattr(config, "RECEPTIONIST_WAKE_PHRASES", "hey reachy,hello reachy")
    phrases = [p.strip() for p in str(wake_raw).split(",") if p.strip()]

    openai_input_hz = 24000

    gate = ReceptionistGate(
        store=store,
        face_embedder=face,
        voice_transcriber=voice,
        camera_worker=camera_worker,
        head_tracker=head_tracker,
        face_threshold=float(getattr(config, "RECEPTIONIST_FACE_THRESHOLD", 0.35)),
        phrase_threshold=float(getattr(config, "RECEPTIONIST_VOICE_THRESHOLD", 0.72)),
        session_ttl_s=float(getattr(config, "RECEPTIONIST_SESSION_TTL_S", 300)),
        buffer_seconds=float(getattr(config, "RECEPTIONIST_AUDIO_BUFFER_S", 6)),
        input_sample_rate=openai_input_hz,
        wake_phrases=phrases,
    )
    logger.info("Receptionist gate initialized (data dir=%s)", root)
    return gate
