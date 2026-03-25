"""Deterministic check-in state machine (Controller subsystem).

Timing budget (total ≤ 3.0 s)
-------------------------------
Stage                        Budget (s)
Face detect + embed + match    0.20
Prompt output                  0.20
Intent window (primary)        1.00   (high-confidence path)
Intent window (fallback)       0.80   (medium-confidence path)
Ledger transaction             0.30
Margin                         ~1.30

Non-negotiable rules
--------------------
* Controller NEVER deducts unless identity AND intent are both confirmed.
* Any failure before ledger commit results in zero state change.
* If cumulative elapsed time > 3.0 s, abort before ledger.
* No cloud calls in any stage.

Failure modes
-------------
no_face_embedding      → idle (face not detected or embed failed)
low_confidence         → fallback (face score below minimum threshold)
no_match               → face + voice do not match any enrolled person
intent_no              → user said no
intent_timeout         → no clear yes/no within the window
time_budget_exceeded   → cumulative time exceeded 3.0 s
insufficient_points    → ledger deduction failed
error                  → unexpected ledger error
"""

from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reachy_mini_conversation_app.receptionist.store import EnrollmentStore, EnrolledPerson
    from reachy_mini_conversation_app.receptionist.face_embed import FaceEmbedder
    from reachy_mini_conversation_app.receptionist.audio_intent import AudioIntentDetector
    from reachy_mini_conversation_app.receptionist.ledger import Ledger
    from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Timing constants
# -----------------------------------------------------------------
BUDGET_TOTAL_S = 3.0
BUDGET_FACE_EMBED_S = 0.20
BUDGET_PROMPT_S = 0.20
BUDGET_INTENT_PRIMARY_S = 1.00   # high-confidence path
BUDGET_INTENT_FALLBACK_S = 0.80  # medium-confidence (voice also checked)
BUDGET_LEDGER_S = 0.30

# Confidence thresholds
CONFIDENCE_HIGH = 0.90   # auto-accept face match, full intent window
CONFIDENCE_MIN = 0.35    # absolute floor — below this → fail immediately


class CheckInState(Enum):
    """States of the check-in state machine."""
    IDLE = auto()
    FACE_EMBED_MATCH = auto()
    FALLBACK_IDENTIFY = auto()
    PROMPT_GREET = auto()
    PROMPT_CHECKIN = auto()
    INTENT_WINDOW = auto()
    LEDGER_DEDUCT = auto()
    SUCCESS = auto()
    FAIL = auto()


@dataclass
class FeedbackEvent:
    """One piece of feedback to send to the robot / UI."""
    state: str
    message: str
    led_state: str          # "blue" | "yellow" | "green" | "red" | "off"
    prompt_for_robot: bool = False   # True → inject this as a spoken prompt


@dataclass
class CheckInResult:
    """Result of one complete check-in attempt."""
    success: bool
    member_id: str | None = None
    display_name: str | None = None
    reason: str = ""
    elapsed_s: float = 0.0
    balance_after: int | None = None
    feedback: list[FeedbackEvent] = field(default_factory=list)


class CheckInController:
    """Deterministic state machine: face embed → intent → ledger.

    This class is thread-safe for calling ``run()`` from a background thread.
    It does NOT use any cloud services.
    """

    def __init__(
        self,
        store: "EnrollmentStore",
        face_embedder: "FaceEmbedder",
        head_tracker: "HeadTracker | None",
        audio_intent: "AudioIntentDetector",
        ledger: "Ledger",
        face_threshold: float = CONFIDENCE_MIN,
        phrase_threshold: float = 0.72,
        check_in_cost: int = 1,
        confidence_high: float = CONFIDENCE_HIGH,
    ) -> None:
        self.store = store
        self.face_embedder = face_embedder
        self.head_tracker = head_tracker
        self.audio_intent = audio_intent
        self.ledger = ledger
        self.face_threshold = face_threshold
        self.phrase_threshold = phrase_threshold
        self.check_in_cost = check_in_cost
        self.confidence_high = confidence_high

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        frame_bgr: np.ndarray,
        gate: object,
    ) -> CheckInResult:
        """Run one complete check-in attempt synchronously.

        Parameters
        ----------
        frame_bgr:
            Latest camera frame (BGR, uint8) — used for face embedding.
        gate:
            The :class:`~reachy_mini_conversation_app.receptionist.gate.ReceptionistGate`
            instance for reading the audio buffer (duck-typed).

        Returns
        -------
        :class:`CheckInResult` with success flag, member info, and feedback events.
        """
        t0 = time.monotonic()
        deadline = t0 + BUDGET_TOTAL_S
        feedback: list[FeedbackEvent] = []

        def elapsed() -> float:
            return time.monotonic() - t0

        def over_budget() -> bool:
            return time.monotonic() > deadline

        # ── Stage 1: Face embed + match (budget 0.20 s) ─────────────────
        logger.debug("Controller: starting face embed+match")
        t_embed = time.monotonic()

        xyxy = None
        if self.head_tracker is not None and hasattr(self.head_tracker, "get_best_face_bbox"):
            xyxy = self.head_tracker.get_best_face_bbox(frame_bgr)

        face_emb = self.face_embedder.embed_frame_with_bbox(frame_bgr, xyxy)

        embed_took = time.monotonic() - t_embed
        logger.debug("Controller: embed took %.3f s (budget %.2f s)", embed_took, BUDGET_FACE_EMBED_S)

        if face_emb is None:
            return CheckInResult(
                success=False,
                reason="no_face_embedding",
                elapsed_s=elapsed(),
                feedback=feedback,
            )

        if over_budget():
            return self._fail("time_budget_exceeded", elapsed(), feedback)

        best_id, best_face_score = self._best_face_match(face_emb)
        logger.debug("Controller: best face match id=%s score=%.3f", best_id, best_face_score)

        # ── Stage 2: Choose path based on confidence ─────────────────────
        person = self.store.get(best_id) if best_id else None

        if best_id is None or best_face_score < self.face_threshold:
            # Below minimum threshold — fail immediately
            feedback.append(FeedbackEvent(
                state="FALLBACK",
                message="I don't recognise you. Please enrol with an administrator.",
                led_state="red",
                prompt_for_robot=True,
            ))
            return CheckInResult(
                success=False,
                reason="low_confidence",
                elapsed_s=elapsed(),
                feedback=feedback,
            )

        if best_face_score >= self.confidence_high:
            # ── HIGH CONFIDENCE PATH ─────────────────────────────────────
            display_name = person.display_name if person else best_id
            feedback.append(FeedbackEvent(
                state="PROMPT_GREET",
                message=f"Hello {display_name}! Say yes to check in.",
                led_state="blue",
                prompt_for_robot=True,
            ))
            intent_window = BUDGET_INTENT_PRIMARY_S

        else:
            # ── MEDIUM CONFIDENCE PATH — also verify voice phrase ────────
            voice_ok = self._verify_voice(gate, person)
            if not voice_ok:
                feedback.append(FeedbackEvent(
                    state="FALLBACK",
                    message="I'm not sure who you are. Please say your passphrase or ask for help.",
                    led_state="red",
                    prompt_for_robot=True,
                ))
                return CheckInResult(
                    success=False,
                    reason="no_match",
                    elapsed_s=elapsed(),
                    feedback=feedback,
                )
            display_name = person.display_name if person else best_id
            feedback.append(FeedbackEvent(
                state="PROMPT_CHECKIN",
                message=f"Welcome back! Say yes to check in.",
                led_state="yellow",
                prompt_for_robot=True,
            ))
            intent_window = BUDGET_INTENT_FALLBACK_S

        if over_budget():
            return self._fail("time_budget_exceeded", elapsed(), feedback)

        # ── Stage 3: Intent window ────────────────────────────────────────
        logger.debug("Controller: waiting %.2f s for intent", intent_window)
        intent = self.audio_intent.detect_from_gate_live(gate, window_seconds=intent_window)
        logger.debug("Controller: intent=%s", intent)

        if intent["confirmed"] != "yes":
            feedback.append(FeedbackEvent(
                state="FAIL",
                message="Check-in cancelled. Come back when you're ready.",
                led_state="off",
            ))
            return CheckInResult(
                success=False,
                reason=f"intent_{intent['confirmed']}",
                elapsed_s=elapsed(),
                feedback=feedback,
            )

        if over_budget():
            return self._fail("time_budget_exceeded", elapsed(), feedback)

        # ── Stage 4: Ledger deduction (budget 0.30 s, only if identity + intent confirmed) ──
        interaction_id = str(uuid.uuid4())
        display_name_safe = person.display_name if person else (best_id or "unknown")

        self.ledger.ensure_member(best_id, display_name_safe)
        t_ledger = time.monotonic()
        ledger_result = self.ledger.deduct(best_id, interaction_id, self.check_in_cost)
        ledger_took = time.monotonic() - t_ledger
        logger.debug("Controller: ledger deduct took %.3f s (budget %.2f s)", ledger_took, BUDGET_LEDGER_S)

        if ledger_result["ok"]:
            balance = self.ledger.get_balance(best_id)
            feedback.append(FeedbackEvent(
                state="SUCCESS",
                message=f"Welcome, {display_name_safe}! Check-in complete. You have {balance} points remaining.",
                led_state="green",
                prompt_for_robot=True,
            ))
            return CheckInResult(
                success=True,
                member_id=best_id,
                display_name=display_name_safe,
                reason="none",
                elapsed_s=elapsed(),
                balance_after=balance,
                feedback=feedback,
            )
        else:
            reason = ledger_result["reason"]
            if reason == "insufficient_points":
                msg = f"Sorry {display_name_safe}, you don't have enough points to check in."
            else:
                msg = "Check-in failed due to a ledger error. Please contact an administrator."
            feedback.append(FeedbackEvent(state="FAIL", message=msg, led_state="red", prompt_for_robot=True))
            return CheckInResult(
                success=False,
                reason=reason,
                elapsed_s=elapsed(),
                feedback=feedback,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_face_match(self, face_emb: np.ndarray) -> tuple[str | None, float]:
        """Find the highest-scoring enrolled face. Returns (person_id, score) or (None, 0.0)."""
        best_id: str | None = None
        best_score = 0.0
        for person in self.store.iter_enrolled():
            score = _cosine(face_emb, person.face_embedding)
            if score > best_score:
                best_score = score
                best_id = person.person_id
        return best_id, best_score

    def _verify_voice(self, gate: object, person: "EnrolledPerson | None") -> bool:
        """Check recent audio buffer against enrolled voice phrase."""
        if person is None:
            return False
        try:
            audio = gate.get_recent_audio()  # type: ignore[attr-defined]
            sr = int(gate.input_sample_rate)  # type: ignore[attr-defined]
            phrase = self.audio_intent.transcriber.transcribe_pcm(audio, sr, min_seconds=0.5)
            if phrase is None:
                return False
            from reachy_mini_conversation_app.receptionist.whisper_voice import phrase_similarity
            score = phrase_similarity(phrase, person.voice_phrase)
            logger.debug("Controller: voice phrase score=%.3f (threshold=%.2f)", score, self.phrase_threshold)
            return score >= self.phrase_threshold
        except Exception as e:
            logger.warning("Controller: voice verify error: %s", e)
            return False

    @staticmethod
    def _fail(reason: str, elapsed: float, feedback: list[FeedbackEvent]) -> CheckInResult:
        feedback.append(FeedbackEvent(state="FAIL", message="Check-in failed. Please try again.", led_state="off"))
        return CheckInResult(success=False, reason=reason, elapsed_s=elapsed, feedback=feedback)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
