"""Wire store, embedders, ledger, audio intent, controller, and gate for receptionist mode."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

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
    camera_worker: "CameraWorker",
    head_tracker: "HeadTracker | None",
) -> ReceptionistGate:
    """Build a :class:`ReceptionistGate` (with ledger + controller) or raise with an actionable error."""
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

    # --- Ledger ----------------------------------------------------------
    ledger = None
    try:
        from reachy_mini_conversation_app.receptionist.ledger import Ledger

        db_path = root / "ledger.db"
        initial_balance = int(getattr(config, "LEDGER_INITIAL_BALANCE", 100))
        ledger = Ledger(db_path=db_path, initial_balance=initial_balance)
        logger.info("Ledger initialised (db=%s, initial_balance=%d)", db_path, initial_balance)
    except Exception as e:
        logger.warning("Ledger init failed — point deductions disabled: %s", e)

    # --- Audio intent detector -------------------------------------------
    audio_intent = None
    try:
        from reachy_mini_conversation_app.receptionist.audio_intent import AudioIntentDetector

        audio_intent = AudioIntentDetector(transcriber=voice)
        logger.info("AudioIntentDetector ready")
    except Exception as e:
        logger.warning("AudioIntentDetector init failed — autonomous check-in will skip intent: %s", e)

    # --- Check-in controller (state machine) -----------------------------
    controller = None
    if ledger is not None and audio_intent is not None:
        try:
            from reachy_mini_conversation_app.receptionist.controller import CheckInController

            face_threshold = float(getattr(config, "RECEPTIONIST_FACE_THRESHOLD", 0.35))
            phrase_threshold = float(getattr(config, "RECEPTIONIST_VOICE_THRESHOLD", 0.72))
            check_in_cost = int(getattr(config, "CHECK_IN_COST_POINTS", 1))
            confidence_high = float(getattr(config, "FACE_CONFIDENCE_HIGH_THRESHOLD", 0.90))

            controller = CheckInController(
                store=store,
                face_embedder=face,
                head_tracker=head_tracker,
                audio_intent=audio_intent,
                ledger=ledger,
                face_threshold=face_threshold,
                phrase_threshold=phrase_threshold,
                check_in_cost=check_in_cost,
                confidence_high=confidence_high,
            )
            logger.info(
                "CheckInController ready (face_threshold=%.2f, phrase_threshold=%.2f, cost=%d, confidence_high=%.2f)",
                face_threshold, phrase_threshold, check_in_cost, confidence_high,
            )
        except Exception as e:
            logger.warning("CheckInController init failed — falling back to LLM-triggered verify: %s", e)

    # --- Wake phrases ----------------------------------------------------
    wake_raw = getattr(config, "RECEPTIONIST_WAKE_PHRASES", "hey reachy,hello reachy,reachy")
    phrases = [p.strip() for p in str(wake_raw).split(",") if p.strip()]

    openai_input_hz = 24_000

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
        controller=controller,
        ledger=ledger,
    )
    logger.info("Receptionist gate initialised (data_dir=%s, autonomous=%s)", root, controller is not None)
    return gate
