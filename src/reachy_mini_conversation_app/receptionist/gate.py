"""Gate OpenAI audio until face+voice verification and a wake phrase.

Two check-in paths
------------------
1. **Autonomous loop** — ``run_autonomous_checkin_loop()`` (async task).
   Watches the camera every 0.5 s; when a face is detected and the gate is
   locked, it spins up the deterministic :class:`CheckInController` in a
   thread.  No LLM involvement.

2. **LLM-triggered** — ``verify_from_buffers()`` (existing, called by the
   ``receptionist_verify`` tool).  Kept for backwards-compat and for cases
   where the admin wants to trigger verification manually.

After either path succeeds, ``unlock()`` is called and the session TTL starts.
The caller must then say a wake phrase before the assistant speaks aloud.
"""

from __future__ import annotations

import time
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.receptionist.store import EnrollmentStore


if TYPE_CHECKING:
    from reachy_mini_conversation_app.camera_worker import CameraWorker
    from reachy_mini_conversation_app.receptionist.face_embed import FaceEmbedder
    from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker
    from reachy_mini_conversation_app.receptionist.whisper_voice import WhisperTranscriber
    from reachy_mini_conversation_app.receptionist.controller import CheckInController
    from reachy_mini_conversation_app.receptionist.ledger import Ledger

logger = logging.getLogger(__name__)

# Seconds between autonomous check-in attempts (cooldown after a fail)
_AUTONOMOUS_COOLDOWN_S = 5.0
# Seconds between camera polls when idle
_POLL_INTERVAL_S = 0.5


class ReceptionistGate:
    """Biometric gate + wake phrase before assistant speaks.

    Parameters
    ----------
    controller:
        Optional :class:`CheckInController` for autonomous check-in.  If
        ``None``, only LLM-triggered verification (``verify_from_buffers``)
        is available.
    ledger:
        Optional :class:`Ledger` reference.  Used by tools that query balances.
    """

    def __init__(
        self,
        store: EnrollmentStore,
        face_embedder: "FaceEmbedder",
        voice_transcriber: "WhisperTranscriber",
        camera_worker: "CameraWorker",
        head_tracker: "HeadTracker | None",
        face_threshold: float,
        phrase_threshold: float,
        session_ttl_s: float,
        buffer_seconds: float,
        input_sample_rate: int,
        wake_phrases: list[str],
        controller: "CheckInController | None" = None,
        ledger: "Ledger | None" = None,
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
        self._audio_lock = threading.Lock()
        self._unlocked_until_mono = 0.0
        self.conversation_live = False
        self.last_unlocked_person: str | None = None

        # Controller (autonomous state machine) and ledger
        self.controller = controller
        self.ledger = ledger

        # Pending spoken prompt for the LLM handler to inject into the realtime session
        self._pending_prompt: str | None = None
        self._pending_prompt_lock = threading.Lock()

        # Dashboard status — updated by autonomous loop and unlock/lock
        self._last_check_status: dict[str, Any] = {
            "state": "IDLE",
            "message": "Waiting for visitor...",
            "led_state": "off",
            "timestamp": time.monotonic(),
            "member_id": None,
            "display_name": None,
        }
        self._status_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Audio buffer
    # ------------------------------------------------------------------

    def record_audio_int16(self, pcm: NDArray[np.int16]) -> None:
        """Append post-resample mono PCM (same rate as ``input_sample_rate``)."""
        flat = pcm.astype(np.int16, copy=False).ravel()
        if flat.size == 0:
            return
        with self._audio_lock:
            self._audio = np.concatenate([self._audio, flat])
            if self._audio.size > self._max_samples:
                self._audio = self._audio[-self._max_samples:]

    def get_recent_audio(self) -> NDArray[np.int16]:
        with self._audio_lock:
            return self._audio.copy()

    def clear_audio_buffer(self) -> None:
        with self._audio_lock:
            self._audio = np.zeros(0, dtype=np.int16)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def is_unlocked(self) -> bool:
        return time.monotonic() < self._unlocked_until_mono

    def get_current_status(self) -> dict[str, Any]:
        """Return a snapshot of the current recognition status for the dashboard."""
        if self.is_unlocked():
            person_id = self.last_unlocked_person
            person = self.store.get(person_id) if person_id else None
            balance = self.ledger.get_balance(person_id) if (self.ledger and person_id) else None
            display_name = person.display_name if person else (person_id or "Unknown")
            return {
                "state": "RECOGNISED",
                "message": f"Access granted — {display_name}",
                "led_state": "green",
                "timestamp": self._unlocked_until_mono - self.session_ttl_s,
                "member_id": person_id,
                "display_name": display_name,
                "roo_points": balance,
                "conversation_live": self.conversation_live,
                "session_expires_in": max(0.0, self._unlocked_until_mono - time.monotonic()),
            }
        with self._status_lock:
            s = dict(self._last_check_status)
        # Add balance if we have a member_id stored
        if s.get("member_id") and self.ledger:
            s["roo_points"] = self.ledger.get_balance(s["member_id"])
        else:
            s["roo_points"] = None
        s["conversation_live"] = False
        s["session_expires_in"] = 0.0
        return s

    def _update_status(self, state: str, message: str, led_state: str, member_id: str | None = None, display_name: str | None = None) -> None:
        """Update the last check status for dashboard display."""
        with self._status_lock:
            self._last_check_status = {
                "state": state,
                "message": message,
                "led_state": led_state,
                "timestamp": time.monotonic(),
                "member_id": member_id,
                "display_name": display_name,
            }

    def unlock(self, person_id: str) -> None:
        self._unlocked_until_mono = time.monotonic() + self.session_ttl_s
        self.conversation_live = False
        self.last_unlocked_person = person_id
        person = self.store.get(person_id)
        display_name = person.display_name if person else person_id
        self._update_status("RECOGNISED", f"Access granted — {display_name}", "green", person_id, display_name)
        logger.info("Receptionist: access session started for %s (wake phrase required)", person_id)

    def lock_session(self) -> None:
        self._unlocked_until_mono = 0.0
        self.conversation_live = False
        self.last_unlocked_person = None
        self._update_status("IDLE", "Waiting for visitor...", "off")
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

    # ------------------------------------------------------------------
    # Pending prompt (for LLM injection)
    # ------------------------------------------------------------------

    def set_pending_prompt(self, text: str) -> None:
        """Store a text prompt to be injected into the OpenAI Realtime session."""
        with self._pending_prompt_lock:
            self._pending_prompt = text

    def pop_pending_prompt(self) -> str | None:
        """Consume and return the pending prompt, or None if none queued."""
        with self._pending_prompt_lock:
            p = self._pending_prompt
            self._pending_prompt = None
        return p

    # ------------------------------------------------------------------
    # Face embedding helper
    # ------------------------------------------------------------------

    def _face_embedding_from_camera(self) -> np.ndarray | None:
        frame = self.camera_worker.get_latest_frame()
        if frame is None:
            return None
        xyxy = None
        if self.head_tracker is not None and hasattr(self.head_tracker, "get_best_face_bbox"):
            xyxy = self.head_tracker.get_best_face_bbox(frame)
        return self.face_embedder.embed_frame_with_bbox(frame, xyxy)

    def _face_present_in_frame(self, frame: np.ndarray) -> bool:
        """Quick check: is there a face in this frame (via YOLO, no InsightFace)."""
        if self.head_tracker is not None and hasattr(self.head_tracker, "get_best_face_bbox"):
            bbox = self.head_tracker.get_best_face_bbox(frame)
            return bbox is not None
        # Fallback: try InsightFace
        emb = self.face_embedder.embed_frame_with_bbox(frame, None)
        return emb is not None

    # ------------------------------------------------------------------
    # LLM-triggered paths (existing API — unchanged)
    # ------------------------------------------------------------------

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

        # Also create ledger entry if ledger is available
        if self.ledger is not None:
            self.ledger.ensure_member(pid, display_name.strip())
            logger.info("Ledger entry created/confirmed for enrolled member %s", pid)

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

        # Ledger deduction via LLM path (cost 1 point)
        ledger_info: dict[str, Any] = {}
        if self.ledger is not None:
            import uuid
            person = self.store.get(pid)
            display_name = person.display_name if person else pid
            self.ledger.ensure_member(pid, display_name)
            deduct_result = self.ledger.deduct(pid, str(uuid.uuid4()), 1)
            ledger_info = {"ledger": deduct_result, "balance": self.ledger.get_balance(pid)}
            if not deduct_result["ok"]:
                return {
                    "ok": False,
                    "error": deduct_result["reason"],
                    "heard_phrase": phrase,
                    "debug": dbg,
                    **ledger_info,
                }

        self.unlock(pid)
        self.clear_audio_buffer()
        return {"ok": True, "person_id": pid, "score": score, "heard_phrase": phrase, "debug": dbg, **ledger_info}

    # ------------------------------------------------------------------
    # Autonomous check-in loop (background async task)
    # ------------------------------------------------------------------

    async def run_autonomous_checkin_loop(self) -> None:
        """Background coroutine: check in visitors automatically when a face is detected.

        - Polls the camera every ``_POLL_INTERVAL_S`` seconds.
        - When a face is present and the gate is locked (no active session),
          runs the :class:`CheckInController` in a thread pool (non-blocking).
        - Applies a cooldown of ``_AUTONOMOUS_COOLDOWN_S`` seconds between
          consecutive attempts (success or fail).
        - After a successful check-in the first ``prompt_for_robot`` message is
          stored as a pending prompt for the LLM handler to speak aloud.
        """
        if self.controller is None:
            logger.warning(
                "Autonomous check-in loop requested but no CheckInController is wired. "
                "The loop will exit. Verification is still available via receptionist_verify."
            )
            return

        logger.info("Receptionist: autonomous check-in loop started")
        last_attempt = 0.0

        while True:
            await asyncio.sleep(_POLL_INTERVAL_S)

            # If already unlocked, nothing to do
            if self.is_unlocked():
                continue

            # Cooldown between attempts
            now = time.monotonic()
            if now - last_attempt < _AUTONOMOUS_COOLDOWN_S:
                continue

            # Quick face-presence check (cheap YOLO call)
            frame = self.camera_worker.get_latest_frame()
            if frame is None:
                continue

            if not self._face_present_in_frame(frame):
                continue

            # Face detected — run the full state machine in a thread
            last_attempt = now
            self._update_status("SCANNING", "Face detected — identifying...", "blue")
            logger.info("Receptionist: face detected, starting autonomous check-in")

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.controller.run,
                    frame,
                    self,
                )
            except Exception as e:
                logger.error("Autonomous check-in raised: %s", e)
                continue

            # Log feedback and forward first robot prompt
            for fb in result.feedback:
                logger.info("Check-in [%s] led=%s: %s", fb.state, fb.led_state, fb.message)
                if fb.prompt_for_robot and self._pending_prompt is None:
                    self.set_pending_prompt(fb.message)

            if result.success:
                self.unlock(result.member_id or "unknown")
                logger.info(
                    "Autonomous check-in SUCCESS: %s (%.2f s, %d pts remaining)",
                    result.display_name,
                    result.elapsed_s,
                    result.balance_after or 0,
                )
            else:
                # Map fail reason to a user-friendly message
                reason_msgs = {
                    "no_face_embedding": "No face detected",
                    "low_confidence": "Face not recognised",
                    "no_match": "Identity not confirmed",
                    "intent_timeout": "No response — check-in cancelled",
                    "intent_no": "Check-in declined",
                    "time_budget_exceeded": "Check-in timed out",
                    "insufficient_points": "Insufficient Roo points",
                    "error": "Ledger error — contact admin",
                }
                msg = reason_msgs.get(result.reason, f"Check-in failed ({result.reason})")
                self._update_status("DENIED", msg, "red")
                logger.info(
                    "Autonomous check-in FAIL: reason=%s (%.2f s)",
                    result.reason,
                    result.elapsed_s,
                )
