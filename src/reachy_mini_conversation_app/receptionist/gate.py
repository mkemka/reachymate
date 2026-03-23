"""Gate OpenAI audio until face+voice verification and a wake phrase."""

from __future__ import annotations
import time
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.receptionist.store import EnrollmentStore


if TYPE_CHECKING:
    from reachy_mini_conversation_app.camera_worker import CameraWorker
    from reachy_mini_conversation_app.receptionist.face_embed import FaceEmbedder
    from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker
    from reachy_mini_conversation_app.receptionist.whisper_voice import WhisperTranscriber


logger = logging.getLogger(__name__)


class ReceptionistGate:
    """Biometric gate + wake phrase before assistant speaks."""

    def __init__(
        self,
        store: EnrollmentStore,
        face_embedder: FaceEmbedder,
        voice_transcriber: WhisperTranscriber,
        camera_worker: CameraWorker,
        head_tracker: HeadTracker | None,
        face_threshold: float,
        phrase_threshold: float,
        session_ttl_s: float,
        buffer_seconds: float,
        input_sample_rate: int,
        wake_phrases: list[str],
    ) -> None:
        self.store = store
        self.face_embedder = face_embedder
        self.voice_transcriber = voice_transcriber
        self.camera_worker = camera_worker
        self.head_tracker = head_tracker
        self.face_threshold = face_threshold
        self.phrase_threshold = phrase_threshold
        self.session_ttl_s = session_ttl_s
        self.wake_phrases = [p.strip().lower() for p in wake_phrases if p.strip()]
        self.input_sample_rate = int(input_sample_rate)
        self._max_samples = max(int(buffer_seconds * input_sample_rate), input_sample_rate // 2)
        self._audio = np.zeros(0, dtype=np.int16)
        self._unlocked_until_mono = 0.0
        self.conversation_live = False
        self.last_unlocked_person: str | None = None

    def record_audio_int16(self, pcm: NDArray[np.int16]) -> None:
        """Append post-resample mono PCM (same rate as ``input_sample_rate``)."""
        flat = pcm.astype(np.int16, copy=False).ravel()
        if flat.size == 0:
            return
        self._audio = np.concatenate([self._audio, flat])
        if self._audio.size > self._max_samples:
            self._audio = self._audio[-self._max_samples :]

    def get_recent_audio(self) -> NDArray[np.int16]:
        return self._audio.copy()

    def clear_audio_buffer(self) -> None:
        self._audio = np.zeros(0, dtype=np.int16)

    def is_unlocked(self) -> bool:
        return time.monotonic() < self._unlocked_until_mono

    def unlock(self, person_id: str) -> None:
        self._unlocked_until_mono = time.monotonic() + self.session_ttl_s
        self.conversation_live = False
        self.last_unlocked_person = person_id
        logger.info("Receptionist: access session started for %s (wake phrase required)", person_id)

    def lock_session(self) -> None:
        self._unlocked_until_mono = 0.0
        self.conversation_live = False
        self.last_unlocked_person = None
        logger.info("Receptionist: session locked")

    def on_user_transcript(self, text: str) -> bool:
        """Return True if wake phrase activated conversation (or already live)."""
        if not self.is_unlocked():
            return False
        if self.conversation_live:
            return True
        lower = text.lower()
        for phrase in self.wake_phrases:
            if phrase in lower:
                self.conversation_live = True
                logger.info("Receptionist: wake phrase matched, assistant audio enabled")
                return True
        return False

    def should_suppress_assistant_output(self) -> bool:
        """Block TTS until face+voice verify succeeded, then until a wake phrase."""
        if not self.is_unlocked():
            return True
        return not self.conversation_live

    def _face_embedding_from_camera(self) -> np.ndarray | None:
        frame = self.camera_worker.get_latest_frame()
        if frame is None:
            return None
        xyxy = None
        if self.head_tracker is not None and hasattr(self.head_tracker, "get_best_face_bbox"):
            xyxy = self.head_tracker.get_best_face_bbox(frame)
        return self.face_embedder.embed_frame_with_bbox(frame, xyxy)

    def enroll_from_buffers(self, display_name: str, replace_person_id: str | None = None) -> dict[str, Any]:
        face_emb = self._face_embedding_from_camera()
        audio = self.get_recent_audio()
        phrase = self.voice_transcriber.transcribe_pcm(audio, self.input_sample_rate, min_seconds=0.8)
        if face_emb is None:
            return {"ok": False, "error": "no_face_embedding", "hint": "Ensure your face is visible to the robot camera."}
        if phrase is None:
            return {
                "ok": False,
                "error": "voice_not_understood",
                "hint": "Speak clearly for 2–4 seconds (your passphrase). Install ffmpeg for Whisper. See github.com/openai/whisper",
            }
        pid = self.store.upsert(display_name, face_emb, phrase, person_id=replace_person_id)
        self.clear_audio_buffer()
        return {"ok": True, "person_id": pid, "display_name": display_name.strip(), "phrase_saved": phrase}

    def verify_from_buffers(self) -> dict[str, Any]:
        face_emb = self._face_embedding_from_camera()
        audio = self.get_recent_audio()
        phrase = self.voice_transcriber.transcribe_pcm(audio, self.input_sample_rate, min_seconds=0.8)
        if face_emb is None:
            return {"ok": False, "error": "no_face_embedding"}
        if phrase is None:
            return {"ok": False, "error": "voice_not_understood"}
        pid, score, dbg = self.store.best_match_dual(
            face_emb,
            phrase,
            self.face_threshold,
            self.phrase_threshold,
        )
        if pid is None:
            return {"ok": False, "error": "no_match", "heard_phrase": phrase, "debug": dbg}
        self.unlock(pid)
        self.clear_audio_buffer()
        return {"ok": True, "person_id": pid, "score": score, "heard_phrase": phrase, "debug": dbg}
